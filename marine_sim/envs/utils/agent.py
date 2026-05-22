import numpy as np
from numpy.linalg import norm
import abc
import logging
from marine_nav.policy.policy_factory import policy_factory
from marine_sim.envs.utils.action import ActionXY, ActionRot
from marine_sim.envs.utils.state import ObservableState, FullState


class Agent(object):
    def __init__(self, config, section):
        """
        Base class for robot and dynamic_obstacle. Have the physical attributes of an agent.

        """
        subconfig = config.robot if section == "robot" else config.dynamic_obstacles
        self.visible = subconfig.visible
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius

        if config.env.randomize_attributes:
            config.orca.neighbor_dist = np.random.uniform(5, 10)
        self.policy = policy_factory[subconfig.policy](config)
        self.sensor = subconfig.sensor
        self.FOV = np.pi * subconfig.FOV

        self.kinematics = (
            "holonomic"
            if section == "dynamic_obstacles"
            else config.action_space.kinematics
        )
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.theta = None
        self.time_step = config.env.time_step
        self.policy.time_step = config.env.time_step

    def print_info(self):
        logging.info(
            "Agent is {} and has {} kinematic constraint".format(
                "visible" if self.visible else "invisible", self.kinematics
            )
        )

    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.5)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta

        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref

    def set_list(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.radius = radius
        self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius)

    def get_observable_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius]

    def get_observable_state_list_noV(self):
        return [self.px, self.py, self.radius]

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == "holonomic":
            next_vx = action.vx
            next_vy = action.vy
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius)

    def get_full_state(self):
        return FullState(
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
        )

    def get_full_state_list(self):
        return [
            self.px,
            self.py,
            self.vx,
            self.vy,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
        ]

    def get_full_state_list_noV(self):
        return [
            self.px,
            self.py,
            self.radius,
            self.gx,
            self.gy,
            self.v_pref,
            self.theta,
        ]

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy

    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]

    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy

        """
        return

    def check_validity(self, action):
        if self.kinematics == "holonomic":
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t, curr_vel=(0, 0)):
        self.check_validity(action)
        if self.kinematics == "holonomic":
            px = self.px + (action.vx + curr_vel[0]) * delta_t
            py = self.py + (action.vy + curr_vel[1]) * delta_t

        else:

            theta = (self.theta + action.r) % (2 * np.pi)
            px = self.px + (np.cos(theta) * action.v + curr_vel[0]) * delta_t
            py = self.py + (np.sin(theta) * action.v + curr_vel[1]) * delta_t

        return px, py

    def step(self, action, curr_vel=(0, 0)):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step, curr_vel)
        self.px, self.py = pos
        if self.kinematics == "holonomic":
            self.vx = action.vx + curr_vel[0]
            self.vy = action.vy + curr_vel[1]
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi)

            self.vx = action.v * np.cos(self.theta) + curr_vel[0]
            self.vy = action.v * np.sin(self.theta) + curr_vel[1]

    def step2(self, action, curr_vel=(0, 0)):
        """
        Perform an action and update the state
        """
        self.vx = self.vx + action.vx
        self.vy = self.vy + action.vy
        vel_norm = np.linalg.norm([self.vx, self.vy])
        if vel_norm > self.v_pref:
            self.vx = self.vx / vel_norm * self.v_pref
            self.vy = self.vy / vel_norm * self.v_pref
        self.theta = np.arctan2(self.vy, self.vx)
        self.px = self.px + (self.vx + curr_vel[0]) * self.time_step
        self.py = self.py + (self.vy + curr_vel[1]) * self.time_step

    def one_step_lookahead(self, pos, action):
        px, py = pos
        self.check_validity(action)
        new_px = px + action.vx * self.time_step
        new_py = py + action.vy * self.time_step
        new_vx = action.vx
        new_vy = action.vy
        return [new_px, new_py, new_vx, new_vy]

    def reached_destination(self):
        return (
            norm(np.array(self.get_position()) - np.array(self.get_goal_position()))
            < self.radius
        )

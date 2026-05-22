import copy
import logging

import gym
import numpy as np
import scipy
from numpy.linalg import norm

from marine_nav.policy.orca import ORCA
from marine_sim.envs.utils.action import ActionRot, ActionXY
from marine_sim.envs.utils.current_flow import (
    COMPUTE_INTERPOLATE_BOX,
    COMPUTE_VELOCITY,
    get_current_flow_vortex_sink,
)
from marine_sim.envs.utils.dynamic_obstacle import DynamicObstacle
from marine_sim.envs.utils.info import *
from marine_sim.envs.utils.robot import Robot
from marine_sim.envs.utils.state import *


class Sonar:

    def __init__(self):
        self.range = 5.0
        self.angle = np.pi
        self.num_beams = 11
        self.compute_phi()
        self.compute_beam_angles()
        self.reflections = []

    def compute_phi(self):
        self.phi = self.angle / (self.num_beams - 1)

    def compute_beam_angles(self):
        self.beam_angles = []
        angle = -self.angle / 2
        for i in range(self.num_beams):
            self.beam_angles.append(angle + i * self.phi)


class Obstacle:

    def __init__(self, x: float, y: float, r: float):
        self.px = x
        self.py = y
        self.radius = r
        self.gx, self.gy = (float("inf"), float("inf"))


class Core:

    def __init__(self, x: float, y: float, clockwise: bool, Gamma: float):
        """
        The class for generatng of a vortex
        :param x: the x-cooridinate of the vortex
        :param y: the y coordinate of the vortex
        :param clockwise: clock wise (positive intesity), counter clockwise (negative intensity)
        :param Gamma: vortex intesity
        """
        self.x = x
        self.y = y
        self.clockwise = clockwise
        self.Gamma = Gamma


class MarineNavigationEnv(gym.Env):
    def __init__(self):
        self.time_limit = None
        self.time_step = None
        self.robot = None
        self.dynamic_obstacles = None
        self.global_time = None
        self.step_counter = 0
        self.goal_arrival_reward = None
        self.collision_penalty = None
        self.encroachment_radius = None
        self.dynamic_obstacle_encroachment_penalty_factor = None
        self.static_obstacle_encroachment_penalty_factor = None
        self.config = None
        self.case_capacity = None
        self.case_size = None
        self.case_counter = None
        self.randomize_attributes = None
        self.circle_radius = None
        self.dynamic_obstacle_num = None
        self.action_space = None
        self.observation_space = None
        self.robot_fov = None
        self.dynamic_obstacle_fov = None
        self.dummy_dynamic_obstacle = None
        self.dummy_robot = None
        self.thisSeed = None
        self.nenv = None
        self.phase = None
        self.test_case = None
        self.render_axis = None
        self.dynamic_obstacles = []
        self.potential = None
        self.desiredVelocity = [0.0, 0.0]
        self.last_left = 0.0
        self.last_right = 0.0
        self.id_counter = None
        self.observed_dynamic_obstacle_ids = None
        self.pred_method = None
        self.pred_method = None
        self.cur_dynamic_obstacle_states = None
        self.rd = np.random.RandomState(self.thisSeed)
        self.r = 0.5
        self.p = 0.8
        self.v_range = [5, 10]
        self.num_cores = 8
        self.cores = []
        self.clear_r = 2.0
        self.v_rel_max = 1.0
        self.sonar = Sonar()
        self.obs_r_range = [0.2, 0.3]
        self.obstacles = []
        self.static_kinematic_dynamic_obstacles = 0
        self.fx = None
        self.fy = None
        self.curr_v, self.eval_curr_v, self.action = (None, None, None)
        self.robot_px, self.robot_py = ([], [])
        self.robot_future_steps = np.zeros(6)

    def _configure_common(self, config):
        self.config = config
        self.time_limit = config.env.time_limit
        self.time_step = config.env.time_step
        self.randomize_attributes = config.env.randomize_attributes
        self.goal_arrival_reward = config.reward.goal_arrival_reward
        self.collision_penalty = config.reward.collision_penalty
        self.encroachment_radius = config.reward.encroachment_radius
        self.dynamic_obstacle_encroachment_penalty_factor = (
            config.reward.dynamic_obstacle_encroachment_penalty_factor
        )
        self.static_obstacle_encroachment_penalty_factor = (
            config.reward.static_obstacle_encroachment_penalty_factor
        )
        self.case_capacity = {
            "train": np.iinfo(np.uint32).max - 2000,
            "val": 1000,
            "test": 1000,
        }
        self.case_size = {
            "train": np.iinfo(np.uint32).max - 2000,
            "val": self.config.env.val_size,
            "test": self.config.env.test_size,
        }
        self.circle_radius = config.sim.circle_radius
        self.dynamic_obstacle_num = config.sim.dynamic_obstacle_num
        self.arena_size = config.sim.arena_size
        self.case_counter = {"train": 0, "test": 0, "val": 0}
        logging.info("dynamic_obstacle number: {}".format(self.dynamic_obstacle_num))
        if self.randomize_attributes:
            logging.info("Randomize dynamic_obstacle's radius and preferred speed")
        else:
            logging.info("Not randomize dynamic_obstacle's radius and preferred speed")
        logging.info("Circle width: {}".format(self.circle_radius))
        self.robot_fov = np.pi * config.robot.FOV
        self.dynamic_obstacle_fov = np.pi * config.dynamic_obstacles.FOV
        logging.info("robot FOV %f", self.robot_fov)
        logging.info("dynamic obstacles FOV %f", self.dynamic_obstacle_fov)
        self.dummy_dynamic_obstacle = DynamicObstacle(self.config, "dynamic_obstacles")
        self.dummy_dynamic_obstacle.set(7, 7, 7, 7, 0, 0, 0)
        self.dummy_dynamic_obstacle.time_step = config.env.time_step
        self.dummy_robot = Robot(self.config, "robot")
        self.dummy_robot.set(7, 7, 7, 7, 0, 0, 0)
        self.dummy_robot.time_step = config.env.time_step
        self.dummy_robot.kinematics = "holonomic"
        self.dummy_robot.policy = ORCA(config)
        self.r = self.config.dynamic_obstacles.radius
        self.random_goal_changing = config.dynamic_obstacles.random_goal_changing
        if self.random_goal_changing:
            self.goal_change_chance = config.dynamic_obstacles.goal_change_chance
        self.end_goal_changing = config.dynamic_obstacles.end_goal_changing
        if self.end_goal_changing:
            self.end_goal_change_chance = (
                config.dynamic_obstacles.end_goal_change_chance
            )
        self.last_dynamic_obstacle_states = np.zeros((self.dynamic_obstacle_num, 5))
        self.predict_steps = config.sim.predict_steps
        self.dynamic_obstacle_num_range = config.sim.dynamic_obstacle_num_range
        assert self.dynamic_obstacle_num > self.dynamic_obstacle_num_range
        self.max_dynamic_obstacle_num = (
            self.dynamic_obstacle_num + self.dynamic_obstacle_num_range
        )
        self.min_dynamic_obstacle_num = (
            self.dynamic_obstacle_num - self.dynamic_obstacle_num_range
        )
        self.use_dummy_detect = False
        self.pred_interval = int(config.data.pred_timestep // config.env.time_step)
        self.buffer_len = self.predict_steps * self.pred_interval
        usv = Robot(config, "robot")
        self.set_robot(usv)

    def _reset_episode_state(self, phase="train", test_case=None):
        self.obstacles.clear()
        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case = self.test_case
        if self.robot is None:
            raise AttributeError("robot has to be set!")
        assert phase in ["train", "val", "test"]
        if test_case is not None:
            self.case_counter[phase] = test_case
        self.global_time = 0
        self.step_counter = 0
        self.id_counter = 0
        self.dynamic_obstacles = []
        self.observed_dynamic_obstacle_ids = []
        counter_offset = {
            "train": self.case_capacity["val"] + self.case_capacity["test"],
            "val": 0,
            "test": self.case_capacity["val"],
        }
        self.rand_seed = (
            counter_offset[phase] + self.case_counter[phase] + self.thisSeed
        )
        np.random.seed(self.rand_seed)
        self.generate_robot_dynamic_obstacles(phase)
        self.cur_dynamic_obstacle_states = np.zeros((self.max_dynamic_obstacle_num, 3))
        for i in range(self.dynamic_obstacle_num):
            self.cur_dynamic_obstacle_states[i] = np.array(
                [
                    self.dynamic_obstacles[i].px,
                    self.dynamic_obstacles[i].py,
                    self.dynamic_obstacles[i].radius,
                ]
            )
        self.case_counter[phase] = (self.case_counter[phase] + int(1 * self.nenv)) % (
            (2 if phase == "test" else 1) * self.case_size[phase]
        )
        usv_goal_vector = np.array([self.robot.gx, self.robot.gy]) - np.array(
            [self.robot.px, self.robot.py]
        )
        self.potential = -abs(np.linalg.norm(usv_goal_vector))
        self.angle = (
            np.arctan2(usv_goal_vector[1], usv_goal_vector[0]) - self.robot.theta
        )
        if self.angle > np.pi:
            self.angle = self.angle - 2 * np.pi
        elif self.angle < -np.pi:
            self.angle = self.angle + 2 * np.pi

    def generate_random_dynamic_obstacle_position(self, dynamic_obstacle_num):
        """
        Calls generate_circle_crossing_dynamic_obstacle function to generate a certain number of random dynamic_obstacles
        :param dynamic_obstacle_num: the total number of dynamic_obstacles to be generated
        :return: None
        """
        for i in range(dynamic_obstacle_num):
            self.dynamic_obstacles.append(
                self.generate_circle_crossing_dynamic_obstacle()
            )

    def update_last_dynamic_obstacle_states(self, dynamic_obstacle_visibility, reset):
        """
        update the self.last_dynamic_obstacle_states array
        dynamic_obstacle_visibility: list of booleans returned by get_dynamic_obstacle_in_fov (e.x. [T, F, F, T, F])
        reset: True if this function is called by reset, False if called by step
        :return:
        """
        for i in range(self.dynamic_obstacle_num):
            if dynamic_obstacle_visibility[i]:
                dynamic_obstacle_state = np.array(
                    self.dynamic_obstacles[i].get_observable_state_list()
                )
                self.last_dynamic_obstacle_states[i, :] = dynamic_obstacle_state
            elif reset:
                dynamic_obstacle_state = np.array([15.0, 15.0, 0.0, 0.0, 0.3])
                self.last_dynamic_obstacle_states[i, :] = dynamic_obstacle_state
            else:
                px, py, vx, vy, r = self.last_dynamic_obstacle_states[i, :]
                px = px + vx * self.time_step
                py = py + vy * self.time_step
                self.last_dynamic_obstacle_states[i, :] = np.array([px, py, vx, vy, r])

    def get_true_dynamic_obstacle_states(self):
        true_dynamic_obstacle_states = np.zeros((self.dynamic_obstacle_num, 2))
        for i in range(self.dynamic_obstacle_num):
            dynamic_obstacle_state = np.array(
                self.dynamic_obstacles[i].get_observable_state_list()
            )
            true_dynamic_obstacle_states[i, :] = dynamic_obstacle_state[:2]
        return true_dynamic_obstacle_states

    def update_dynamic_obstacle_goals_randomly(self):
        for dynamic_obstacle in self.dynamic_obstacles:
            if dynamic_obstacle.isObstacle or dynamic_obstacle.v_pref == 0:
                continue
            if np.random.random() > self.goal_change_chance:
                continue

            dynamic_obstacles_copy = [
                obstacle
                for obstacle in self.dynamic_obstacles
                if obstacle != dynamic_obstacle
            ]

            while True:
                angle = np.random.random() * np.pi * 2
                v_pref = (
                    1.0 if dynamic_obstacle.v_pref == 0 else dynamic_obstacle.v_pref
                )
                gx_noise = (np.random.random() - 0.5) * v_pref
                gy_noise = (np.random.random() - 0.5) * v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False

                for agent in [self.robot] + dynamic_obstacles_copy:
                    min_dist = (
                        dynamic_obstacle.radius
                        + agent.radius
                        + self.encroachment_radius
                    )
                    if (
                        norm((gx - agent.px, gy - agent.py)) < min_dist
                        or norm((gx - agent.gx, gy - agent.gy)) < min_dist
                    ):
                        collide = True
                        break
                if not collide:
                    break

            dynamic_obstacle.gx = gx
            dynamic_obstacle.gy = gy

    def update_dynamic_obstacle_goal(self, dynamic_obstacle):
        if np.random.random() <= self.end_goal_change_chance:
            dynamic_obstacles_copy = []
            for h in self.dynamic_obstacles:
                if h != dynamic_obstacle:
                    dynamic_obstacles_copy.append(h)
            while True:
                angle = np.random.random() * np.pi * 2
                v_pref = (
                    1.0 if dynamic_obstacle.v_pref == 0 else dynamic_obstacle.v_pref
                )
                gx_noise = (np.random.random() - 0.5) * v_pref
                gy_noise = (np.random.random() - 0.5) * v_pref
                gx = self.circle_radius * np.cos(angle) + gx_noise
                gy = self.circle_radius * np.sin(angle) + gy_noise
                collide = False
                for agent in [self.robot] + dynamic_obstacles_copy:
                    min_dist = (
                        dynamic_obstacle.radius
                        + agent.radius
                        + self.encroachment_radius
                    )
                    if (
                        norm((gx - agent.px, gy - agent.py)) < min_dist
                        or norm((gx - agent.gx, gy - agent.gy)) < min_dist
                    ):
                        collide = True
                        break
                if not collide:
                    break
            dynamic_obstacle.gx = gx
            dynamic_obstacle.gy = gy
        return

    def calc_offset_angle(self, state1, state2):
        if self.robot.kinematics == "holonomic":
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        v_fov = [np.cos(real_theta), np.sin(real_theta)]
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)
        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        return offset

    def detect_visible(
        self, state1, state2, robot1=False, custom_fov=None, custom_sensor_range=None
    ):
        if self.robot.kinematics == "holonomic":
            real_theta = np.arctan2(state1.vy, state1.vx)
        else:
            real_theta = state1.theta
        v_fov = [np.cos(real_theta), np.sin(real_theta)]
        v_12 = [state2.px - state1.px, state2.py - state1.py]
        v_fov = v_fov / np.linalg.norm(v_fov)
        v_12 = v_12 / np.linalg.norm(v_12)
        offset = np.arccos(np.clip(np.dot(v_fov, v_12), a_min=-1, a_max=1))
        if custom_fov:
            fov = custom_fov
        elif robot1:
            fov = self.robot_fov
        else:
            fov = self.dynamic_obstacle_fov
        if np.abs(offset) <= fov / 2:
            inFov = True
        else:
            inFov = False
        dist = (
            np.linalg.norm([state1.px - state2.px, state1.py - state2.py])
            - state1.radius
            - state2.radius
        )
        if custom_sensor_range:
            inSensorRange = dist <= custom_sensor_range
        elif robot1:
            inSensorRange = dist <= self.robot.sensor_range
        else:
            inSensorRange = True
        return inFov and inSensorRange

    def last_dynamic_obstacle_states_obj(self):
        """
        convert self.last_dynamic_obstacle_states to a list of observable state objects for old algorithms to use
        """
        dynamic_obstacles = []
        for i in range(self.dynamic_obstacle_num):
            h = ObservableState(*self.last_dynamic_obstacle_states[i])
            dynamic_obstacles.append(h)
        return dynamic_obstacles

    def assign_df(self, df):
        self.df = df

    def rereset_from_pandas(self, df):
        import pandas as pd
        from marine_nav.policy.policy_factory import policy_factory
        from .utils.current_flow import get_current_flow_vortex_sink

        self.rand_seed = df["rand_seed"]
        np.random.seed(self.rand_seed)
        self.start = df["start"]
        self.goal = df["goal"]
        self.robot.set(
            df["robot"]["px"],
            df["robot"]["py"],
            df["robot"]["gx"],
            df["robot"]["gy"],
            0,
            0,
            np.pi / 2,
        )
        self.robot.prev_velocity = ActionXY(vx=df["robot"]["vx"], vy=df["robot"]["vy"])
        self.robot_px.clear()
        self.robot_py.clear()
        self.gamma, self.X0, self.Y0 = ([], [], [])
        self.cores = []
        for core in df["cores"]:
            self.cores.append(
                Core(core["x"], core["y"], core["clockwise"], core["Gamma"])
            )
        for core in self.cores:
            if core.clockwise:
                self.gamma.append(core.Gamma)
            else:
                self.gamma.append(-core.Gamma)
            self.X0.append(core.x)
            self.Y0.append(core.y)
        gamma, X0, Y0 = (self.gamma, self.X0, self.Y0)
        self.num_cores = len(self.cores)
        self.free_velocity = df["free_velocity"]
        self.alpha = df["alpha"]
        self.fx, self.fy = get_current_flow_vortex_sink(
            gamma=gamma[: self.num_cores // 2],
            X0_V=X0[: self.num_cores // 2],
            Y0_V=Y0[: self.num_cores // 2],
            lamda=gamma[self.num_cores // 2 :],
            X0_S=X0[self.num_cores // 2 :],
            Y0_S=Y0[self.num_cores // 2 :],
            Vinf=self.free_velocity,
            alpha=self.alpha,
            XL=-self.width,
            XR=self.width,
            YL=-self.height,
            YR=self.height,
        )
        self.obstacle_positions, self.obstacles = ([], [])
        self.obstacle_num = len(df["static_obstacles"])
        for obs in df["static_obstacles"]:
            px, py, radius = (obs["px"], obs["py"], obs["radius"])
            self.obstacles.append(Obstacle(px, py, radius))
            self.obstacle_positions.append([px, py])
        self.obstacle_positions = np.array(self.obstacle_positions)
        self.dynamic_obstacles = []
        for i, hu in enumerate(df["dynamic_obstacles"]):
            dynamic_obstacle = DynamicObstacle(self.config, "dynamic_obstacles")
            if self.randomize_attributes:
                dynamic_obstacle.sample_random_attributes()
            dynamic_obstacle.id = i
            dynamic_obstacle.radius = hu["radius"]
            px, py = (hu["px"], hu["py"])
            dynamic_obstacle.set(px, py, -px, -py, 0, 0, 0)
            dynamic_obstacle.visible = hu["visible"]
            dynamic_obstacle.v_pref = hu["v_pref"]
            dynamic_obstacle.policy = policy_factory[hu["policy"]["name"].lower()](
                self.config
            )
            dynamic_obstacle.sensor = hu["sensor"]
            dynamic_obstacle.FOV = hu["FOV"]
            dynamic_obstacle.kinematics = hu["kinematics"]
            dynamic_obstacle.time_step = hu["time_step"]
            dynamic_obstacle.policy.time_step = hu["policy"]["time_step"]
            self.dynamic_obstacles.append(dynamic_obstacle)
        self.dynamic_obstacle_num = len(df["dynamic_obstacles"])
        self.last_dynamic_obstacle_states = np.zeros((self.dynamic_obstacle_num, 5))
        self.cur_dynamic_obstacle_states = np.zeros((self.max_dynamic_obstacle_num, 3))
        for i in range(self.dynamic_obstacle_num):
            self.cur_dynamic_obstacle_states[i] = np.array(
                [
                    self.dynamic_obstacles[i].px,
                    self.dynamic_obstacles[i].py,
                    self.dynamic_obstacles[i].radius,
                ]
            )
        phase = "test"
        usv_goal_vector = np.array([self.robot.gx, self.robot.gy]) - np.array(
            [self.robot.px, self.robot.py]
        )
        self.potential = -abs(np.linalg.norm(usv_goal_vector))
        self.angle = (
            np.arctan2(usv_goal_vector[1], usv_goal_vector[0]) - self.robot.theta
        )
        if self.angle > np.pi:
            self.angle = self.angle - 2 * np.pi
        elif self.angle < -np.pi:
            self.angle = self.angle + 2 * np.pi

    def generate_robot_dynamic_obstacles(self, phase, dynamic_obstacle_num=None):
        if self.robot.kinematics == "unicycle":
            angle = np.random.uniform(0, np.pi * 2)
            px = self.arena_size * np.cos(angle)
            py = self.arena_size * np.sin(angle)
            while True:
                gx, gy = np.random.uniform(-self.arena_size, self.arena_size, 2)
                if np.linalg.norm([px - gx, py - gy]) >= 12:
                    break
            self.robot.set(px, py, gx, gy, 0, 0, np.random.uniform(0, 2 * np.pi))
            self.dynamic_obstacle_num = np.random.randint(
                1,
                self.config.sim.spawned_dynamic_obstacle_num
                + self.dynamic_obstacle_num_range
                + 1,
            )
        else:
            while True:
                px, py, gx, gy = np.random.uniform(
                    -self.arena_size, self.arena_size, 4
                )
                if np.linalg.norm([px - gx, py - gy]) >= 8:
                    break
            self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
            self.robot.set(px, py, gx, gy, 0, 0, np.pi / 2)
            print(
                f"[marine_env INFO] Generate {self.spawned_dynamic_obstacle_num} dynamic obstacles"
            )
            self.dynamic_obstacle_num = np.random.randint(
                low=self.config.sim.spawned_dynamic_obstacle_num
                - self.dynamic_obstacle_num_range,
                high=self.config.sim.spawned_dynamic_obstacle_num
                + self.dynamic_obstacle_num_range
                + 1,
            )
        self.generate_random_dynamic_obstacle_position(
            dynamic_obstacle_num=self.dynamic_obstacle_num
        )
        self.last_dynamic_obstacle_states = np.zeros((self.dynamic_obstacle_num, 5))
        for i in range(self.dynamic_obstacle_num):
            self.dynamic_obstacles[i].id = i

    def calc_dynamic_obstacle_future_traj(self, method):
        dynamic_obstacle_num = (
            self.dynamic_obstacle_num + 1
            if self.robot.visible
            else self.dynamic_obstacle_num
        )
        if method == "truth":
            self.dynamic_obstacle_future_traj = np.zeros(
                (self.buffer_len + 1, dynamic_obstacle_num, 4)
            )
        elif method == "const_vel":
            self.dynamic_obstacle_future_traj = np.zeros(
                (self.predict_steps + 1, dynamic_obstacle_num, 4)
            )
        else:
            raise NotImplementedError
        for i in range(self.dynamic_obstacle_num):
            self.dynamic_obstacle_future_traj[0, i] = np.array(
                self.dynamic_obstacles[i].get_observable_state_list()[:-1]
            )
        if method == "const_vel":
            self.dynamic_obstacle_future_traj[0, :, 2:4] = (
                self.prev_dynamic_obstacle_pos[:, 2:4]
            )
        if self.robot.visible:
            self.dynamic_obstacle_future_traj[0, -1] = np.array(
                self.robot.get_observable_state_list()[:-1]
            )
        if method == "truth":
            for i in range(1, self.buffer_len + 1):
                for j in range(self.dynamic_obstacle_num):
                    full_state = np.concatenate(
                        (
                            self.dynamic_obstacle_future_traj[i - 1, j],
                            self.dynamic_obstacles[j].get_full_state_list()[4:],
                        )
                    )
                    observable_states = []
                    for k in range(self.dynamic_obstacle_num):
                        if j == k:
                            continue
                        observable_states.append(
                            np.concatenate(
                                (
                                    self.dynamic_obstacle_future_traj[i - 1, k],
                                    [self.dynamic_obstacles[k].radius],
                                )
                            )
                        )
                    action = self.dynamic_obstacles[j].act_joint_state(
                        JointState(full_state, observable_states)
                    )
                    self.dynamic_obstacle_future_traj[i, j] = self.dynamic_obstacles[
                        j
                    ].one_step_lookahead(
                        self.dynamic_obstacle_future_traj[i - 1, j, :2], action
                    )
                if self.robot.visible:
                    action = ActionXY(*self.dynamic_obstacle_future_traj[i - 1, -1, 2:])
                    self.dynamic_obstacle_future_traj[i, -1] = (
                        self.robot.one_step_lookahead(
                            self.dynamic_obstacle_future_traj[i - 1, -1, :2], action
                        )
                    )
            self.dynamic_obstacle_future_traj = self.dynamic_obstacle_future_traj[
                :: self.pred_interval
            ]
        elif method == "const_vel":
            self.dynamic_obstacle_future_traj = np.tile(
                self.dynamic_obstacle_future_traj[0].reshape(
                    1, dynamic_obstacle_num, 4
                ),
                (self.predict_steps + 1, 1, 1),
            )
            pred_timestep = np.tile(
                np.arange(0, self.predict_steps + 1, dtype=float).reshape(
                    (self.predict_steps + 1, 1, 1)
                )
                * self.time_step
                * self.pred_interval,
                [1, dynamic_obstacle_num, 2],
            )
            pred_disp = pred_timestep * self.dynamic_obstacle_future_traj[:, :, 2:]
            self.dynamic_obstacle_future_traj[:, :, :2] = (
                self.dynamic_obstacle_future_traj[:, :, :2] + pred_disp
            )
        else:
            raise NotImplementedError
        if self.robot.visible:
            self.dynamic_obstacle_future_traj = self.dynamic_obstacle_future_traj[
                :, :-1
            ]
        self.dynamic_obstacle_future_traj[
            :, np.logical_not(self.dynamic_obstacle_visibility), :2
        ] = 15
        self.dynamic_obstacle_future_traj[
            :, np.logical_not(self.dynamic_obstacle_visibility), 2:
        ] = 0
        return self.dynamic_obstacle_future_traj

    def update_dynamic_obstacle_pos_goal(self, dynamic_obstacle):
        while True:
            angle = np.random.random() * np.pi * 2
            v_pref = 1.0 if dynamic_obstacle.v_pref == 0 else dynamic_obstacle.v_pref
            gx_noise = (np.random.random() - 0.5) * v_pref
            gy_noise = (np.random.random() - 0.5) * v_pref
            gx = self.circle_radius * np.cos(angle) + gx_noise
            gy = self.circle_radius * np.sin(angle) + gy_noise
            collide = False
            if not collide:
                break
        dynamic_obstacle.gx = gx
        dynamic_obstacle.gy = gy

    def configure(self, config):
        self.obstacle_num = config.sim.obstacle_num
        config.obstacles = self.obstacles
        " read the config to the environment variables "
        self._configure_common(config)
        self.sonar.range = self.robot.sensor_range
        self.pred_method = config.sim.predict_method
        self.width = self.circle_radius / np.sqrt(2)
        self.height = self.circle_radius / np.sqrt(2)
        self.spawned_dynamic_obstacle_num = config.sim.spawned_dynamic_obstacle_num
        self.spawned_obstacle_num = config.sim.spawned_obstacle_num

    def reset(self, phase="train", test_case=None):
        self._reset_episode_state(phase=phase, test_case=test_case)
        self.start = np.array([self.robot.px, self.robot.py])
        self.goal = [self.robot.gx, self.robot.gy]
        self.robot.prev_velocity = ActionXY(vx=self.robot.vx, vy=self.robot.vy)
        self.generate_reset_cores()
        self.robot_px.clear()
        self.robot_py.clear()
        gamma, X0, Y0 = ([], [], [])
        for core in self.cores:
            if core.clockwise:
                gamma.append(core.Gamma)
            else:
                gamma.append(-core.Gamma)
            X0.append(core.x)
            Y0.append(core.y)
        self.free_velocity = 1
        self.alpha = np.random.randint(low=0, high=45)
        self.fx, self.fy = get_current_flow_vortex_sink(
            gamma=gamma[: self.num_cores // 2],
            X0_V=X0[: self.num_cores // 2],
            Y0_V=Y0[: self.num_cores // 2],
            lamda=gamma[self.num_cores // 2 :],
            X0_S=X0[self.num_cores // 2 :],
            Y0_S=Y0[self.num_cores // 2 :],
            Vinf=self.free_velocity,
            alpha=self.alpha,
            XL=-self.width,
            XR=self.width,
            YL=-self.height,
            YR=self.height,
        )
        self.Vx, self.Vy = (None, None)
        self.curr_v, self.eval_curr_v = (None, None)
        ob = self.generate_ob(reset=True, sort=self.config.args.sort_dynamic_obstacles)
        return ob

    def generate_reset_cores(self):
        self.cores.clear()
        num_cores = self.num_cores
        if num_cores > 0:
            iteration = 500
            while True:
                center = self.rd.uniform(
                    low=np.array([-self.width, -self.height]),
                    high=np.array([self.width, self.height]),
                )
                direction = self.rd.binomial(1, 0.5)
                v_edge = self.rd.uniform(low=self.v_range[0], high=self.v_range[1])
                Gamma = 2 * np.pi * self.r * v_edge
                core = Core(center[0], center[1], direction, Gamma)
                iteration -= 1
                if self.check_core(core):
                    self.cores.append(core)
                    num_cores -= 1
                if iteration == 0 or num_cores == 0:
                    break
        centers = None
        for core in self.cores:
            if centers is None:
                centers = np.array([[core.x, core.y]])
            else:
                c = np.array([[core.x, core.y]])
                centers = np.vstack((centers, c))
        if centers is not None:
            self.core_centers = scipy.spatial.KDTree(centers)

    def set_robot(self, robot):
        self.robot = robot
        d = {}
        d["ego_state"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 7), dtype=np.float32
        )
        d["ego_velocity"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 2), dtype=np.float32
        )
        d["dynamic_obstacle_states"] = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                self.config.sim.dynamic_obstacle_num
                + self.config.sim.dynamic_obstacle_num_range,
                int(2 * (self.predict_steps + 1)),
            ),
            dtype=np.float32,
        )
        d["detected_dynamic_obstacle_num"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        grid_size = self.config.robot.flow_grid_num**2
        d["current_flow"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2, grid_size), dtype=np.float32
        )
        d["static_obstacle_states"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11, 3), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(d)
        high = np.inf * np.ones([2])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)

    def generate_obstacles(self):
        self.obstacle_positions = []
        obstacle_scale_factor = 1.0 if self.obstacle_num == 10 else 5.0
        for i in range(self.spawned_obstacle_num):
            radius = self.rd.uniform(
                low=obstacle_scale_factor * self.obs_r_range[0],
                high=obstacle_scale_factor * self.obs_r_range[1],
            )
            px, py = self.generate_circle_crossing_dynamic_obstacle(
                radius=radius, return_pos=True, offset_dist=5
            )
            self.obstacles.append(Obstacle(px, py, radius))
            self.obstacle_positions.append([px, py])
        self.obstacle_positions = np.array(self.obstacle_positions)

    def generate_circle_crossing_dynamic_obstacle(
        self, radius=0.3, return_pos=False, offset_dist=1.5
    ):
        if not return_pos and self.obstacle_num != 0 and (len(self.obstacles) == 0):
            self.generate_obstacles()
            self.config.obstacles = self.obstacles
        if not return_pos:
            dynamic_obstacle = DynamicObstacle(self.config, "dynamic_obstacles")
            if self.randomize_attributes:
                dynamic_obstacle.sample_random_attributes()
            radius = dynamic_obstacle.radius
        while True:
            if return_pos:
                px, py = np.random.uniform(-self.arena_size, self.arena_size, 2)
            else:
                angle = np.random.random() * np.pi * 2
                noise_range = 2
                px_noise = np.random.uniform(0, 1) * noise_range
                py_noise = np.random.uniform(0, 1) * noise_range
                px = self.circle_radius * np.cos(angle) + px_noise
                py = self.circle_radius * np.sin(angle) + py_noise
            collide = False
            for i, agent in enumerate(
                [self.robot] + self.obstacles + self.dynamic_obstacles
            ):
                if self.robot.kinematics == "unicycle" and i == 0:
                    min_dist = self.circle_radius / 2
                else:
                    min_dist = (
                        radius + agent.radius + self.encroachment_radius + offset_dist
                    )
                if (
                    norm((px - agent.px, py - agent.py)) < min_dist
                    or norm((px - agent.gx, py - agent.gy)) < min_dist
                ):
                    collide = True
                    break
            if not collide:
                break
        if not return_pos:
            dynamic_obstacle.set(px, py, -px, -py, 0, 0, 0)
            return dynamic_obstacle
        return (px, py)

    def generate_ob(self, reset, sort=True):
        """Generate observation for reset and step functions"""
        ob = {}
        visible_dynamic_obstacles, num_visibles, self.dynamic_obstacle_visibility = (
            self.get_num_dynamic_obstacle_in_fov()
        )
        ob["ego_state"] = self.robot.get_full_state_list_noV()
        self.prev_dynamic_obstacle_pos = copy.deepcopy(
            self.last_dynamic_obstacle_states
        )
        self.update_last_dynamic_obstacle_states(
            self.dynamic_obstacle_visibility, reset=reset
        )
        ob["ego_velocity"] = np.array([self.robot.vx, self.robot.vy])
        ob["dynamic_obstacle_states"] = (
            np.ones(
                (
                    self.config.sim.dynamic_obstacle_num
                    + self.config.sim.dynamic_obstacle_num_range,
                    int(2 * (self.predict_steps + 1)),
                )
            )
            * np.inf
        )
        predicted_states = self.calc_dynamic_obstacle_future_traj(
            method=self.pred_method
        )
        predicted_pos = np.transpose(predicted_states[:, :, :2], (1, 0, 2)) - np.array(
            [self.robot.px, self.robot.py]
        )
        ob["dynamic_obstacle_states"][: self.dynamic_obstacle_num][
            self.dynamic_obstacle_visibility
        ] = predicted_pos.reshape((self.dynamic_obstacle_num, -1))[
            self.dynamic_obstacle_visibility
        ]
        if self.config.args.sort_dynamic_obstacles:
            ob["dynamic_obstacle_states"] = np.array(
                sorted(
                    ob["dynamic_obstacle_states"],
                    key=lambda x: np.linalg.norm(x[:2]),
                )
            )
        ob["dynamic_obstacle_states"][np.isinf(ob["dynamic_obstacle_states"])] = 15
        ob["detected_dynamic_obstacle_num"] = num_visibles
        if ob["detected_dynamic_obstacle_num"] == 0:
            ob["detected_dynamic_obstacle_num"] = 1
        Rrw = self.get_robot_transform(self.robot)
        grid_size = self.config.robot.flow_grid_num**2
        ob["current_flow"] = np.zeros((2, grid_size))
        ob["static_obstacle_states"] = np.zeros((11, 3))
        if not reset:
            _, _, Vx, Vy = COMPUTE_INTERPOLATE_BOX(
                self.robot.px,
                self.robot.py,
                self.fx,
                self.fy,
                a=self.config.robot.flow_a,
                b=self.config.robot.flow_b,
                num=self.config.robot.flow_grid_num,
            )
            V = np.stack((Vx, Vy))
            V = Rrw.dot(V)
            ob["current_flow"] = np.tanh(V)
            self.measure_static_obstacles()
            static_obstacle_measurements = np.stack(self.sonar.reflections)
            self.static_obstacle_min_distance = np.min(
                np.linalg.norm(static_obstacle_measurements[:, :2], axis=1)
            )
            ob["static_obstacle_states"] = static_obstacle_measurements
        return ob

    def get_dynamic_obstacle_actions(self):
        dynamic_obstacle_actions = []
        for i in range(
            self.static_kinematic_dynamic_obstacles, self.dynamic_obstacle_num
        ):
            dynamic_obstacle = self.dynamic_obstacles[i]
            ob = []
            for other_dynamic_obstacle in self.dynamic_obstacles:
                if other_dynamic_obstacle != dynamic_obstacle:
                    if self.detect_visible(dynamic_obstacle, other_dynamic_obstacle):
                        ob.append(other_dynamic_obstacle.get_observable_state())
                    else:
                        ob.append(self.dummy_dynamic_obstacle.get_observable_state())
            if self.robot.visible:
                if self.detect_visible(self.dynamic_obstacles[i], self.robot):
                    ob += [self.robot.get_observable_state()]
                else:
                    ob += [self.dummy_robot.get_observable_state()]
            dynamic_obstacle_actions.append(dynamic_obstacle.act(ob))
        return dynamic_obstacle_actions

    def get_num_dynamic_obstacle_in_fov(self):
        dynamic_obstacle_ids = []
        dynamic_obstacles_in_view = []
        num_dynamic_obstacles_in_view = 0
        for i in range(self.dynamic_obstacle_num):
            visible = self.detect_visible(
                self.robot, self.dynamic_obstacles[i], robot1=True
            )
            if visible:
                dynamic_obstacles_in_view.append(self.dynamic_obstacles[i])
                num_dynamic_obstacles_in_view = num_dynamic_obstacles_in_view + 1
                dynamic_obstacle_ids.append(True)
            else:
                dynamic_obstacle_ids.append(False)
        return (
            dynamic_obstacles_in_view,
            num_dynamic_obstacles_in_view,
            dynamic_obstacle_ids,
        )

    def get_robot_transform(self, robot):
        theta = np.arctan2(robot.vy, robot.vx)
        R_rw = np.matrix(
            [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        )
        return R_rw

    def step(self, action, update=True):
        """
        step function
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        self.rew_thres = 0.2
        if self.robot.policy.name == "ORCA":
            _, _, dynamic_obstacle_visibility = self.get_num_dynamic_obstacle_in_fov()
            dynamic_obstacle_states = copy.deepcopy(
                self.calc_dynamic_obstacle_future_traj(method="truth")
            )
            dynamic_obstacle_states = np.concatenate(
                (
                    dynamic_obstacle_states.reshape((-1, 4)),
                    np.tile(
                        self.last_dynamic_obstacle_states[:, -1],
                        self.predict_steps + 1,
                    ).reshape((-1, 1)),
                ),
                axis=1,
            )
            action = self.robot.act(dynamic_obstacle_states.tolist())
        elif self.robot.kinematics == "holonomic":
            deltaV = self.robot.policy.clip_action(action, 0.4)
            action = np.array(
                [
                    self.robot.prev_velocity.vx + deltaV.vx,
                    self.robot.prev_velocity.vy + deltaV.vy,
                ]
            )
            action = self.robot.policy.clip_action(action, self.robot.v_pref)
            self.robot.prev_velocity = action
        else:
            action = self.robot.policy.clip_action(action, 0.1)
        self.action = action
        if self.robot.kinematics == "unicycle":
            kinematic_viscosity_resistance = (
                0.1 / self.robot.v_pref * self.desiredVelocity[0]
            )
            self.desiredVelocity[0] = np.clip(
                self.desiredVelocity[0] + action.v - kinematic_viscosity_resistance,
                -0.5 * self.robot.v_pref,
                self.robot.v_pref,
            )
            action = ActionRot(self.desiredVelocity[0], action.r)
            action = ActionRot(self.desiredVelocity[0], action.r)
        dynamic_obstacle_actions = self.get_dynamic_obstacle_actions()
        if self.phase == "test":
            self.calc_dynamic_obstacle_future_traj(method="truth")
        reward, done, episode_info = self.compute_total_reward(
            action, danger_zone="future"
        )
        curr_vx, curr_vy = self.get_velocity(self.robot.px, self.robot.py)
        self.curr_v = (curr_vx, curr_vy)
        self.robot.step(action, (curr_vx, curr_vy))
        for i, dynamic_obstacle_action in enumerate(dynamic_obstacle_actions):
            self.dynamic_obstacles[i].step(dynamic_obstacle_action)
            self.cur_dynamic_obstacle_states[i] = np.array(
                [
                    self.dynamic_obstacles[i].px,
                    self.dynamic_obstacles[i].py,
                    self.dynamic_obstacles[i].radius,
                ]
            )
        self.global_time += self.time_step
        self.step_counter = self.step_counter + 1
        info = {"info": episode_info}
        ob = self.generate_ob(reset=False)
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_dynamic_obstacle_goals_randomly()
        if self.end_goal_changing:
            for i in range(
                self.static_kinematic_dynamic_obstacles, self.dynamic_obstacle_num
            ):
                dynamic_obstacle = self.dynamic_obstacles[i]
                if (
                    norm(
                        (
                            dynamic_obstacle.gx - dynamic_obstacle.px,
                            dynamic_obstacle.gy - dynamic_obstacle.py,
                        )
                    )
                    < dynamic_obstacle.radius
                ):
                    self.dynamic_obstacles[i] = (
                        self.generate_circle_crossing_dynamic_obstacle()
                    )
                    self.dynamic_obstacles[i].id = i
        return (ob, reward, done, info)

    def current_flow_stagnation_penalty(self):
        pot_factor = -0.2
        curr_flow_speed = COMPUTE_VELOCITY(
            self.robot.px, self.robot.py, self.robot.theta, self.fx, self.fy
        )
        usv_desired_speed = self.desiredVelocity[0]
        reward = 0
        if self.robot.v_pref < curr_flow_speed:
            reward = pot_factor * max(curr_flow_speed - usv_desired_speed, 0)
        return reward

    def base_navigation_reward(self, action, danger_zone="circle"):
        dmin = float("inf")
        collision = False
        closest_obstacle_type = None

        for agent in self.obstacles:
            dx = agent.px - self.robot.px
            dy = agent.py - self.robot.py
            closest_dist = (dx**2 + dy**2) ** (1 / 2) - agent.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist
                closest_obstacle_type = "static"

        for agent in self.dynamic_obstacles:
            dx = agent.px - self.robot.px
            dy = agent.py - self.robot.py
            closest_dist = (dx**2 + dy**2) ** (1 / 2) - agent.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist
                closest_obstacle_type = "dynamic"

        if self.robot.kinematics == "unicycle":
            goal_radius = 0.6
        else:
            goal_radius = self.robot.radius
        reaching_goal = (
            norm(
                np.array(self.robot.get_position())
                - np.array(self.robot.get_goal_position())
            )
            < goal_radius
        )
        if danger_zone == "circle" or self.phase == "train":
            danger_cond = dmin < self.encroachment_radius
            min_danger_dist = 0
        else:
            relative_pos = self.dynamic_obstacle_future_traj[1:, :, :2] - np.array(
                [self.robot.px, self.robot.py]
            )
            relative_dist = np.linalg.norm(relative_pos, axis=-1)
            collision_idx = (
                relative_dist < self.robot.radius + self.config.dynamic_obstacles.radius
            )
            danger_cond = np.any(collision_idx)
            if danger_cond:
                min_danger_dist = np.amin(relative_dist[collision_idx])
            else:
                min_danger_dist = 0
        if self.global_time >= self.time_limit - 1:
            reward = 0
            done = True
            episode_info = Timeout()
        elif collision:
            reward = self.collision_penalty
            done = True
            episode_info = Collision()
        elif reaching_goal:
            reward = self.goal_arrival_reward
            done = True
            episode_info = ReachGoal()
        elif danger_cond:
            if closest_obstacle_type == "static":
                encroachment_penalty_factor = (
                    self.static_obstacle_encroachment_penalty_factor
                )
            else:
                encroachment_penalty_factor = (
                    self.dynamic_obstacle_encroachment_penalty_factor
                )
            reward = (
                (dmin - self.encroachment_radius)
                * encroachment_penalty_factor
                * self.time_step
            )
            done = False
            episode_info = Danger(min_danger_dist)
        else:
            if self.robot.kinematics == "holonomic":
                pot_factor = 2
            else:
                pot_factor = 3.0
            potential_cur = np.linalg.norm(
                np.array([self.robot.px, self.robot.py])
                - np.array(self.robot.get_goal_position())
            )
            potential_reward = pot_factor * (-abs(potential_cur) - self.potential)
            reward = potential_reward
            self.potential = -abs(potential_cur)
            if reward < self.rew_thres:
                reward -= 0.2
            done = False
            episode_info = Nothing()
        r_spin, r_back = (0, 0)
        if self.robot.kinematics == "unicycle":
            if action.v < 0:
                r_back = -2 * abs(action.v)
            else:
                r_back = 0.0
            reward = reward + r_spin + r_back
        return (reward, done, episode_info)

    def static_obstacle_encroachment_penalty(self):
        self.predict_ego_steps = 3
        self.robot_future_steps = np.stack(
            (
                self.robot.px
                + np.arange(1, self.predict_ego_steps + 1)
                * self.robot.vx
                * self.time_step,
                self.robot.py
                + np.arange(1, self.predict_ego_steps + 1)
                * self.robot.vy
                * self.time_step,
            ),
            axis=1,
        )
        collision_idx = (
            np.linalg.norm(
                self.obstacle_positions - self.robot_future_steps.reshape(-1, 1, 2),
                axis=-1,
            )
            < self.robot.radius + 2.0
        )
        coefficients = 2.0 ** np.arange(2, self.predict_ego_steps + 2).reshape(
            (self.predict_ego_steps, 1)
        )
        collision_penalties = self.collision_penalty / coefficients
        static_obstacle_encroachment_penalty = collision_idx * collision_penalties
        static_obstacle_encroachment_penalty = np.min(
            static_obstacle_encroachment_penalty
        )
        return static_obstacle_encroachment_penalty

    def compute_total_reward(self, action, danger_zone="future"):
        base_reward, done, episode_info = self.base_navigation_reward(
            action, danger_zone=danger_zone
        )
        static_obstacle_encroachment_penalty = (
            self.static_obstacle_encroachment_penalty()
        )
        relative_pos = self.dynamic_obstacle_future_traj[1:, :, :2] - np.array(
            [self.robot.px, self.robot.py]
        )
        collision_idx = (
            np.linalg.norm(relative_pos, axis=-1)
            < self.robot.radius + self.config.dynamic_obstacles.radius
        )
        coefficients = 2.0 ** np.arange(2, self.predict_steps + 2).reshape(
            (self.predict_steps, 1)
        )
        collision_penalties = self.collision_penalty / coefficients
        dynamic_obstacle_encroachment_penalty = collision_idx * collision_penalties
        dynamic_obstacle_encroachment_penalty = np.min(
            dynamic_obstacle_encroachment_penalty
        )
        total_reward = (
            base_reward
            + dynamic_obstacle_encroachment_penalty
            + static_obstacle_encroachment_penalty
        )
        self.reward = total_reward
        return (total_reward, done, episode_info)

    def check_core(self, core_j):
        if core_j.x - self.r < -self.width or core_j.x + self.r > self.width:
            return False
        if core_j.y - self.r < -self.height or core_j.y + self.r > self.height:
            return False
        core_pos = np.array([core_j.x, core_j.y])
        dis_s = core_pos - self.start
        if np.linalg.norm(dis_s) < self.r + self.clear_r:
            return False
        dis_g = core_pos - self.goal
        if np.linalg.norm(dis_g) < self.r + self.clear_r:
            return False
        for core_i in self.cores:
            dx = core_i.x - core_j.x
            dy = core_i.y - core_j.y
            dis = np.sqrt(dx * dx + dy * dy)
            if core_i.clockwise == core_j.clockwise:
                boundary_i = core_i.Gamma / (2 * np.pi * self.v_rel_max)
                boundary_j = core_j.Gamma / (2 * np.pi * self.v_rel_max)
                if dis < boundary_i + boundary_j:
                    return False
            else:
                Gamma_l = max(core_i.Gamma, core_j.Gamma)
                Gamma_s = min(core_i.Gamma, core_j.Gamma)
                v_1 = Gamma_l / (2 * np.pi * (dis - 2 * self.r))
                v_2 = Gamma_s / (2 * np.pi * self.r)
                if v_1 > self.p * v_2:
                    return False
        return True

    def get_velocity(self, x: float, y: float):
        return (float(self.fx.ev(y, x)), float(self.fy.ev(y, x)))

    def measure_static_obstacles(self):
        px, py = (self.robot.px, self.robot.py)
        theta = self.robot.theta
        self.sonar.reflections.clear()
        for rel_a in self.sonar.beam_angles:
            angle = theta + rel_a
            angle = angle % (2 * np.pi)
            reflection_dist = np.inf
            w = np.array([2 * self.sonar.range, 2 * self.sonar.range, -1.0])
            obstacle_px, obstacle_py = (
                2 * self.sonar.range + px,
                2 * self.sonar.range + py,
            )
            for obs in self.obstacles:
                if not self.detect_visible(
                    self.robot,
                    obs,
                    robot1=True,
                    custom_fov=self.sonar.angle,
                    custom_sensor_range=self.sonar.range,
                ):
                    continue
                if (
                    np.abs(angle - np.pi / 2) < 0.001
                    or np.abs(angle - 3 * np.pi / 2) < 0.001
                ):
                    M = obs.radius * obs.radius - (px - obs.px) * (px - obs.px)
                    if M < 0.0:
                        continue
                    x1 = px
                    x2 = px
                    y1 = obs.py - np.sqrt(M)
                    y2 = obs.py + np.sqrt(M)
                else:
                    K = np.tan(angle)
                    a = 1 + K * K
                    b = 2 * K * (py - K * px - obs.py) - 2 * obs.px
                    c = (
                        obs.px * obs.px
                        + (py - K * px - obs.py) * (py - K * px - obs.py)
                        - obs.radius * obs.radius
                    )
                    delta = b * b - 4 * a * c
                    if delta < 0.0:
                        continue
                    x1 = (-b - np.sqrt(delta)) / (2 * a)
                    x2 = (-b + np.sqrt(delta)) / (2 * a)
                    y1 = py + K * (x1 - px)
                    y2 = py + K * (x2 - px)
                v1 = np.array([x1 - px, y1 - py])
                v2 = np.array([x2 - px, y2 - py])
                v = v1 if np.linalg.norm(v1) < np.linalg.norm(v2) else v2
                if np.linalg.norm(v) > self.sonar.range:
                    continue
                if np.dot(v, np.array([np.cos(angle), np.sin(angle)])) < 0.0:
                    continue
                if np.linalg.norm(v) >= reflection_dist:
                    continue
                else:
                    w = np.array([np.abs(v[0]), np.abs(v[1]), angle])
                    reflection_dist = np.linalg.norm(v)
            self.sonar.reflections.append(w)
        return self.sonar.reflections

    def render_trajectories(
        self,
        robot_px=[],
        robot_py=[],
        traj_colors=[],
        dynamic_obstacle_pos=[],
        legends=[],
        step_count=None,
    ):
        """Render the current status of the environment using matplotlib"""
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
        self.robot_px, self.robot_py = (robot_px, robot_py)
        robot_color = "gold"
        goal_color = "red"
        arrow_color = "red"
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            FOVLineRot = np.array(
                [
                    [np.cos(ang), -np.sin(ang), 0],
                    [np.sin(ang), np.cos(ang), 0],
                    [0, 0, 1],
                ]
            )
            point.extend([1])
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        ax = self.render_axis
        artists = []

        def render_flow(XX, YY, Vx, Vy):
            M = np.hypot(Vx, Vy)
            stride_size = 3
            field_arrow = plt.quiver(
                XX[::stride_size, ::stride_size],
                YY[::stride_size, ::stride_size],
                Vx[::stride_size, ::stride_size],
                Vy[::stride_size, ::stride_size],
                scale=1,
                scale_units="xy",
                angles="xy",
                alpha=0.7,
            )
            return field_arrow

        if not self.Vx is not None:
            X = np.linspace(*ax.get_xlim(), 100)
            Y = np.linspace(*ax.get_ylim(), 100)
            self.XX, self.YY = np.meshgrid(X, Y)
            self.Vx = self.fx.ev(self.YY, self.XX)
            self.Vy = self.fy.ev(self.YY, self.XX)
        field_arrow = render_flow(self.XX, self.YY, self.Vx, self.Vy)
        artists.append(field_arrow)
        goal = mlines.Line2D(
            [self.robot.gx],
            [self.robot.gy],
            color=goal_color,
            marker="*",
            linestyle="None",
            markersize=15,
        )
        ax.add_artist(goal)
        artists.append(goal)
        robotX, robotY = self.start
        robot = plt.Circle((robotX, robotY), 1, fill=True, color=robot_color)
        ax.add_artist(robot)
        artists.append(robot)
        if self.robot.FOV < 2 * np.pi:
            FOVAng = self.robot_fov / 2
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle="--")
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle="--")
            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)
            FOVEndPoint1 = calcFOVLineEndPoint(
                FOVAng,
                [endPointX - startPointX, endPointY - startPointY],
                20.0 / self.robot.radius,
            )
            FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            FOVEndPoint2 = calcFOVLineEndPoint(
                -FOVAng,
                [endPointX - startPointX, endPointY - startPointY],
                20.0 / self.robot.radius,
            )
            FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))
            ax.add_artist(FOVLine1)
            ax.add_artist(FOVLine2)
            artists.append(FOVLine1)
            artists.append(FOVLine2)
        if not self.curr_v == None:
            from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox, HPacker

            texts = [
                "currX = ",
                "currY = ",
                "steerV = ",
                "steerTheta = ",
                "VX = ",
                "VY = ",
                "action-r = ",
                "action-theta = ",
                "static_obstacle_min_distance = ",
                "reward = ",
            ]
            if not self.eval_curr_v == None:
                assert np.allclose(self.eval_curr_v, self.curr_v)
            self.eval_curr_v = (
                self.fx.ev(self.robot.py, self.robot.px).item(),
                self.fy.ev(self.robot.py, self.robot.px).item(),
            )
            values = [
                *self.eval_curr_v,
                self.desiredVelocity[0],
                self.robot.theta,
                self.robot.vx,
                self.robot.vy,
                *self.action,
                self.static_obstacle_min_distance,
                self.reward,
            ]
            Texts = []
            for t, v in zip(texts, values):
                Texts.append(TextArea(t + f"{v:.2f}", textprops=dict(color="blue")))
            texts_vbox = VPacker(children=Texts, pad=0, sep=0)
            text = AnnotationBbox(texts_vbox, self.robot.get_position(), frameon=False)
            ax.add_artist(text)
            artists.append(text)

        def obs_get_position(obs):
            return (obs.px, obs.py)

        dynamic_obstacle_radius = self.dynamic_obstacles[0].radius
        dynamic_obstacle_circles = [
            plt.Circle(
                dynamic_obstacle,
                dynamic_obstacle_radius,
                fill=False,
                linewidth=1.5,
                label=None,
            )
            for i, dynamic_obstacle in enumerate(dynamic_obstacle_pos)
        ]
        static_obstacles_marks = [
            plt.Circle(
                (obs.px, obs.py),
                obs.radius,
                fill=True,
                color="gray",
                linewidth=1.5,
                label=None,
            )
            for i, obs in enumerate(self.obstacles)
        ]
        actual_arena_size = self.arena_size + 0.5
        for i in range(len(dynamic_obstacle_pos)):
            ax.add_artist(dynamic_obstacle_circles[i])
            artists.append(dynamic_obstacle_circles[i])
            dynamic_obstacle_circles[i].set_color(c="r")
            if (
                -actual_arena_size <= dynamic_obstacle_pos[i][0] <= actual_arena_size
                and -actual_arena_size
                <= dynamic_obstacle_pos[i][1]
                <= actual_arena_size
            ):
                plt.text(
                    dynamic_obstacle_pos[i][0] - 0.1,
                    dynamic_obstacle_pos[i][1] - 0.1,
                    i,
                    color="black",
                    fontsize=12,
                )
        for i in range(len(self.obstacles)):
            ax.add_artist(static_obstacles_marks[i])
            artists.append(static_obstacles_marks[i])
            static_obstacles_marks[i].set_color(c="#341c02")
        for traj_px, traj_py, traj_color, legend in zip(
            robot_px, robot_py, traj_colors, legends
        ):
            traj = mlines.Line2D(
                traj_px,
                traj_py,
                color=traj_color,
                linestyle="--",
                linewidth=4,
                markersize=25,
                label=legend,
            )
            ax.add_artist(traj)
            artists.append(traj)
        if self.config.save_slides:
            import os

            if self.config.save_path.endswith("/"):
                self.config.save_path = self.config.save_path[:-1]
            folder_path = os.path.join(self.config.save_path + "_Traj")
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            if step_count == 1671:
                ax_legend = ax.legend(
                    fontsize=20, loc="upper left", prop={"weight": "bold", "size": 20}
                )
                ax_legend.get_frame().set_alpha(0.5)
            elif ax.legend_:
                ax.legend_.remove()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    folder_path,
                    str(step_count if step_count else self.rand_seed) + "_traj.png",
                ),
                bbox_inches="tight",
                dpi=300,
            )
        plt.pause(0.1)
        artists = artists
        for item in artists:
            item.remove()
        for t in ax.texts:
            t.set_visible(False)
        return ax

    def render(self):
        """Render the current status of the environment using matplotlib"""
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches

        plt.rcParams["animation.ffmpeg_path"] = "/usr/bin/ffmpeg"
        robot_color = "gold"
        goal_color = "red"
        arrow_color = "red"
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        def calcFOVLineEndPoint(ang, point, extendFactor):
            FOVLineRot = np.array(
                [
                    [np.cos(ang), -np.sin(ang), 0],
                    [np.sin(ang), np.cos(ang), 0],
                    [0, 0, 1],
                ]
            )
            point.extend([1])
            newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
            newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
            return newPoint

        ax = self.render_axis
        artists = []

        def render_flow(XX, YY, Vx, Vy):
            M = np.hypot(Vx, Vy)
            stride_size = 3
            field_arrow = plt.quiver(
                XX[::stride_size, ::stride_size],
                YY[::stride_size, ::stride_size],
                Vx[::stride_size, ::stride_size],
                Vy[::stride_size, ::stride_size],
                scale=1,
                scale_units="xy",
                angles="xy",
            )
            return field_arrow

        if not self.Vx is not None:
            X = np.linspace(*ax.get_xlim(), 100)
            Y = np.linspace(*ax.get_ylim(), 100)
            self.XX, self.YY = np.meshgrid(X, Y)
            self.Vx = self.fx.ev(self.YY, self.XX)
            self.Vy = self.fy.ev(self.YY, self.XX)
        field_arrow = render_flow(self.XX, self.YY, self.Vx, self.Vy)
        artists.append(field_arrow)
        goal = mlines.Line2D(
            [self.robot.gx],
            [self.robot.gy],
            color=goal_color,
            marker="*",
            linestyle="None",
            markersize=15,
            label="Goal",
        )
        ax.add_artist(goal)
        artists.append(goal)
        start_loc = plt.Circle(
            self.start, 1, fill=True, color=robot_color, label="Start"
        )
        ax.add_artist(start_loc)
        artists.append(start_loc)
        robotX, robotY = self.robot.get_position()
        robot = plt.Circle(
            (robotX, robotY), self.robot.radius, fill=True, color=robot_color
        )
        ax.add_artist(robot)
        artists.append(robot)
        radius = self.robot.radius
        arrowStartEnd = []
        robot_theta = (
            self.robot.theta
            if self.robot.kinematics == "unicycle"
            else np.arctan2(self.robot.vy, self.robot.vx)
        )
        arrowStartEnd.append(
            (
                (robotX, robotY),
                (
                    robotX + radius * np.cos(robot_theta),
                    robotY + radius * np.sin(robot_theta),
                ),
            )
        )
        for i, dynamic_obstacle in enumerate(self.dynamic_obstacles):
            theta = np.arctan2(dynamic_obstacle.vy, dynamic_obstacle.vx)
            arrowStartEnd.append(
                (
                    (dynamic_obstacle.px, dynamic_obstacle.py),
                    (
                        dynamic_obstacle.px + radius * np.cos(theta),
                        dynamic_obstacle.py + radius * np.sin(theta),
                    ),
                )
            )
        arrows = [
            patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
            for arrow in arrowStartEnd
        ]
        for arrow in arrows:
            ax.add_artist(arrow)
            artists.append(arrow)
        if self.sonar.angle < 2 * np.pi:
            FOVAngBeams = self.sonar.beam_angles
            FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle="--")
            FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle="--")
            startPointX = robotX
            startPointY = robotY
            endPointX = robotX + radius * np.cos(robot_theta)
            endPointY = robotY + radius * np.sin(robot_theta)
            FOVLineList = []
            for FOVAng in FOVAngBeams:
                FOVEndPoint1 = calcFOVLineEndPoint(
                    FOVAng,
                    [endPointX - startPointX, endPointY - startPointY],
                    self.sonar.range / radius,
                )
                FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle="--", c="#06c2ac")
                FOVLine1.set_xdata(
                    np.array([startPointX, startPointX + FOVEndPoint1[0]])
                )
                FOVLine1.set_ydata(
                    np.array([startPointY, startPointY + FOVEndPoint1[1]])
                )
                FOVLineList.append(FOVLine1)
            for i in range(len(FOVLineList)):
                ax.add_artist(FOVLineList[i])
                artists.append(FOVLineList[i])
        if not self.curr_v == None:
            from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox, HPacker

            texts = [
                "currX = ",
                "currY = ",
                "steerV = ",
                "steerTheta = ",
                "VX = ",
                "VY = ",
                "action-r = ",
                "action-theta = ",
                "static_obstacle_min_distance = ",
                "reward = ",
            ]
            if not self.eval_curr_v == None:
                assert np.allclose(self.eval_curr_v, self.curr_v)
            self.eval_curr_v = (
                self.fx.ev(self.robot.py, self.robot.px).item(),
                self.fy.ev(self.robot.py, self.robot.px).item(),
            )
            values = [
                *self.eval_curr_v,
                self.desiredVelocity[0],
                self.robot.theta,
                self.robot.vx,
                self.robot.vy,
                *self.action,
                self.static_obstacle_min_distance,
                self.reward,
            ]
            Texts = []
            for t, v in zip(texts, values):
                Texts.append(TextArea(t + f"{v:.2f}", textprops=dict(color="blue")))
        sensor_range = plt.Circle(
            self.robot.get_position(),
            self.robot.sensor_range
            + self.robot.radius
            + self.config.dynamic_obstacles.radius,
            fill=True,
            alpha=0.5,
            color="pink",
            linestyle="--",
        )
        ax.add_artist(sensor_range)
        artists.append(sensor_range)

        def obs_get_position(obs):
            return (obs.px - obs.radius / 2, obs.py - obs.radius / 2)

        dynamic_obstacle_circles = [
            plt.Circle(
                dynamic_obstacle.get_position(),
                dynamic_obstacle.radius,
                fill=False,
                linewidth=1.5,
                label="DO" if i == 0 else "",
            )
            for i, dynamic_obstacle in enumerate(self.dynamic_obstacles)
        ]
        static_obstacles_marks = [
            plt.Circle(
                (obs.px, obs.py),
                obs.radius,
                fill=True,
                color="gray",
                linewidth=1.5,
                label="SO" if i == 0 else "",
            )
            for i, obs in enumerate(self.obstacles)
        ]
        for i in range(len(self.obstacles)):
            ax.add_artist(static_obstacles_marks[i])
            artists.append(static_obstacles_marks[i])
            if self.detect_visible(
                self.robot,
                self.obstacles[i],
                custom_fov=self.sonar.angle,
                custom_sensor_range=self.sonar.range,
                robot1=True,
            ):
                static_obstacles_marks[i].set_color(c="#06c2ac")
            else:
                static_obstacles_marks[i].set_color(c="#341c02")
        actual_arena_size = self.arena_size + 0.5
        for i in range(len(self.dynamic_obstacles)):
            ax.add_artist(dynamic_obstacle_circles[i])
            artists.append(dynamic_obstacle_circles[i])
            if self.detect_visible(self.robot, self.dynamic_obstacles[i], robot1=True):
                dynamic_obstacle_circles[i].set_color(c="pink")
            else:
                dynamic_obstacle_circles[i].set_color(c="r")
        if self.config.save_slides:
            import os

            folder_path = os.path.join(
                self.config.save_path, str(self.rand_seed) + "pred"
            )
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path, exist_ok=True)
            plt.savefig(
                os.path.join(folder_path, str(self.step_counter) + ".png"), dpi=300
            )
        plt.pause(0.1)
        for item in artists:
            item.remove()
        for t in ax.texts:
            t.set_visible(False)

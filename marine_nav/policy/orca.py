import numpy as np
import rvo2
from marine_nav.policy.policy import Policy
from marine_sim.envs.utils.action import ActionXY


class ORCA(Policy):
    def __init__(self, config):
        """
        timeStep        The time step of the simulation.
                        Must be positive.
        neighborDist    The default maximum distance (center point
                        to center point) to other agents a new agent
                        takes into account in the navigation. The
                        larger this number, the longer the running
                        time of the simulation. If the number is too
                        low, the simulation will not be safe. Must be
                        non-negative.
        maxNeighbors    The default maximum number of other agents a
                        new agent takes into account in the
                        navigation. The larger this number, the
                        longer the running time of the simulation.
                        If the number is too low, the simulation
                        will not be safe.
        timeHorizon     The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        other agents. The larger this number, the
                        sooner an agent will respond to the presence
                        of other agents, but the less freedom the
                        agent has in choosing its velocities.
                        Must be positive.
        timeHorizonObst The default minimal amount of time for which
                        a new agent's velocities that are computed
                        by the simulation are safe with respect to
                        obstacles. The larger this number, the
                        sooner an agent will respond to the presence
                        of obstacles, but the less freedom the agent
                        has in choosing its velocities.
                        Must be positive.
        radius          The default radius of a new agent.
                        Must be non-negative.
        maxSpeed        The default maximum speed of a new agent.
                        Must be non-negative.
        velocity        The default initial two-dimensional linear
                        velocity of a new agent (optional).

        ORCA first uses neighborDist and maxNeighbors to find neighbors that need to be taken into account.
        Here set them to be large enough so that all agents will be considered as neighbors.
        Time_horizon should be set that at least it's safe for one time step

        In this work, obstacles are not considered. So the value of time_horizon_obst doesn't matter.

        """
        super().__init__(config)
        self.name = "ORCA"
        self.max_neighbors = None
        self.radius = None
        self.max_speed = 1
        self.sim = None
        self.safety_space = self.config.orca.safety_space

    def predict(self, state):
        """
        Create a rvo2 simulation at each time step and run one step
        Python-RVO2 API: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/rvo2.pyx
        How simulation is done in RVO2: https://github.com/sybrenstuvel/Python-RVO2/blob/master/src/Agent.cpp

        Agent doesn't stop moving after it reaches the goal, because once it stops moving, the reciprocal rule is broken

        :param state:
        :return:
        """
        self_state = state.self_state

        self.max_neighbors = len(state.dynamic_obstacle_states)
        self.radius = state.self_state.radius
        params = (
            self.config.orca.neighbor_dist,
            self.max_neighbors,
            self.config.orca.time_horizon,
            self.config.orca.time_horizon_obst,
        )
        if (
            self.sim is not None
            and self.sim.getNumAgents() != len(state.dynamic_obstacle_states) + 1
        ):
            del self.sim
            self.sim = None
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(
                self.time_step, *params, self.radius, self.max_speed
            )
            self.sim.addAgent(
                (self_state.px, self_state.py),
                *params,
                self_state.radius + 0.01 + self.safety_space,
                self_state.v_pref,
                (self_state.vx, self_state.vy)
            )
            for dynamic_obstacle_state in state.dynamic_obstacle_states:
                self.sim.addAgent(
                    (dynamic_obstacle_state.px, dynamic_obstacle_state.py),
                    *params,
                    dynamic_obstacle_state.radius
                    + 0.01
                    + self.config.orca.safety_space,
                    self.max_speed,
                    (dynamic_obstacle_state.vx, dynamic_obstacle_state.vy)
                )
        else:
            self.sim.setAgentPosition(0, (self_state.px, self_state.py))
            self.sim.setAgentVelocity(0, (self_state.vx, self_state.vy))
            for i, dynamic_obstacle_state in enumerate(state.dynamic_obstacle_states):
                self.sim.setAgentPosition(
                    i + 1, (dynamic_obstacle_state.px, dynamic_obstacle_state.py)
                )
                self.sim.setAgentVelocity(
                    i + 1, (dynamic_obstacle_state.vx, dynamic_obstacle_state.vy)
                )

        velocity = np.array(
            (self_state.gx - self_state.px, self_state.gy - self_state.py)
        )
        speed = np.linalg.norm(velocity)
        pref_vel = velocity / speed if speed > 1 else velocity

        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))
        for i, dynamic_obstacle_state in enumerate(state.dynamic_obstacle_states):

            self.sim.setAgentPrefVelocity(i + 1, (0, 0))

        self.sim.doStep()
        action = ActionXY(*self.sim.getAgentVelocity(0))
        self.last_state = state

        return action

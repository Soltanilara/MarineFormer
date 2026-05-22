from marine_sim.envs.utils.agent import Agent
from marine_sim.envs.utils.state import JointState


class DynamicObstacle(Agent):

    def __init__(self, config, section):
        super().__init__(config, section)
        self.isObstacle = False
        self.id = None
        self.observed_id = -1

    def act(self, ob):
        """
        The state for dynamic_obstacle is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action

    def act_joint_state(self, ob):
        """
        The state for dynamic_obstacle is its full state and all other agents' observable states
        :param ob:
        :return:
        """
        action = self.policy.predict(ob)
        return action

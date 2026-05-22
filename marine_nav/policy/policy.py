import abc
import numpy as np


class Policy(object):
    def __init__(self, config):
        """
        Base class for all policies, has an abstract method predict().
        """
        self.trainable = False
        self.phase = None
        self.model = None
        self.device = None
        self.last_state = None
        self.time_step = None

        self.env = None
        self.config = config

    @abc.abstractmethod
    def predict(self, state):
        """
        Policy takes state as input and output an action

        """
        return

    @staticmethod
    def reach_destination(state):
        self_state = state.self_state
        if (
            np.linalg.norm(
                (self_state.py - self_state.gy, self_state.px - self_state.gx)
            )
            < self_state.radius
        ):
            return True
        else:
            return False


class NeuralPolicy(Policy):
    def __init__(self, config):
        super().__init__(config)
        self.name = "marineformer"
        self.trainable = True
        self.multiagent_training = True

    def clip_action(self, raw_action, v_pref):
        if self.config.action_space.kinematics == "holonomic":
            act_norm = np.linalg.norm(raw_action)
            if act_norm > v_pref:
                raw_action[0] = raw_action[0] / act_norm * v_pref
                raw_action[1] = raw_action[1] / act_norm * v_pref
            from marine_sim.envs.utils.action import ActionXY

            return ActionXY(raw_action[0], raw_action[1])

        from marine_sim.envs.utils.action import ActionRot

        act_norm = np.clip(raw_action[0], -v_pref, v_pref)
        theta = np.clip(raw_action[1], -0.1, 0.1)
        return ActionRot(act_norm, theta)

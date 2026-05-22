policy_factory = dict()


def none_policy():
    return None


from marine_nav.policy.orca import ORCA
from marine_nav.policy.orca_obstacles import ORCA_Obstacles
from marine_nav.policy.policy import NeuralPolicy

policy_factory["orca"] = ORCA
policy_factory["orca_obstacles"] = ORCA_Obstacles
policy_factory["none"] = none_policy
policy_factory["marineformer"] = NeuralPolicy

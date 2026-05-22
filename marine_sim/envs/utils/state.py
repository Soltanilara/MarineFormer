from collections import namedtuple
import numpy as np

FullState = namedtuple(
    "FullState", ["px", "py", "vx", "vy", "radius", "gx", "gy", "v_pref", "theta"]
)
ObservableState = namedtuple("ObservableState", ["px", "py", "vx", "vy", "radius"])


class JointState(object):

    def __init__(self, self_state, dynamic_obstacle_states):
        assert len(self_state) == 9
        dynamic_obstacle_states_namedtuple = []

        if len(np.shape(dynamic_obstacle_states)) == 2:
            for dynamic_obstacle_state in dynamic_obstacle_states:
                assert len(dynamic_obstacle_state) == 5
                dynamic_obstacle_states_namedtuple.append(
                    ObservableState(*dynamic_obstacle_state)
                )

        else:
            assert len(dynamic_obstacle_states) % 5 == 0
            dynamic_obstacle_num = len(dynamic_obstacle_states) // 5
            for i in range(dynamic_obstacle_num):
                dynamic_obstacle_states_namedtuple.append(
                    ObservableState(
                        *dynamic_obstacle_states[int(i * 5) : (int((i + 1) * 5))]
                    )
                )

        self.self_state = FullState(*self_state)
        self.dynamic_obstacle_states = dynamic_obstacle_states_namedtuple

    def to_flatten_list(self):
        flatten_list = list(self.self_state)
        for dynamic_obstacle_state in self.dynamic_obstacle_states:
            flatten_list.extend(list(dynamic_obstacle_state))
        return flatten_list

import collections

basic_transition = collections.namedtuple("Transition",
                                          ["current_state", "action", "reward", "next_state", "terminal",
                                           "mission", "mission_length", "gamma"])
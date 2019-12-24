import collections
import operator
import random
import numpy as np

from abc import ABC
from abc import abstractmethod

class AbstractReplay(ABC):
    def __init__(self, size, seed, hindsight_reward):
        self.transition = collections.namedtuple("Transition",
                                                 ["curr_state", "action", "reward", "next_state", "terminal",
                                                  "mission", "mission_length"])
        self.stored_transitions = []
        self.current_episode = []
        self.hindsight_reward = hindsight_reward

        self.memory_size = int(size)
        self.memory = [None for _ in range(self.memory_size)]
        self.len = 0
        self.position = 0
        random.seed(seed)

    @abstractmethod
    def add_transition(self, curr_state, action, reward, next_state, terminal, mission, mission_length, hindsight_mission):
        pass
    @abstractmethod
    def sample(self):
        pass
    def update_weight(self):
        pass

class ReplayMemory(AbstractReplay):
    def __init__(self, size, seed, hindsight_reward):
        super().__init__(self, size, seed, hindsight_reward)

    def add_transition(self, curr_state, action, reward, next_state, terminal, mission, mission_length, hindsight_mission=None):
        """
        Adds transition to an temporary episode buffer
        When TERMINAL is reach, the episode is added to the replay buffer

        This allows to switch mission and reward for the whole episode, if necessary.

        If hindsight mission is provided (terminal == True is needed)
        The current episode is stored alongside a new one with the substitued mission
        """
        # assert mission_length == mission.size(1), "Mission length doesn't match, 'mission_length' in state"
        # mission = mission[0] if mission.shape[0] == 0 else mission

        self.current_episode.append(
            self.transition(curr_state, action, reward, next_state, terminal, mission, mission_length)
        )

        if hindsight_mission:
            assert terminal is True, "If hindsight mission is provided, should be at the end of episode"

            # Substitute the old mission with the new one, change the reward too
            hindsight_episode = [self.transition(st, a, self.hindsight_reward, st_plus1, end_ep, hindsight_mission,
                                                 len(hindsight_mission))
                                 for st, a, r, st_plus1, end_ep, wrong_mission, length in self.current_episode
                                 ]
            self.store_episode(hindsight_episode)

        if terminal:
            self.store_episode(self.current_episode)
            self.current_episode = []

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def store_episode(self, episode_to_store):

        len_episode = len(episode_to_store)
        if self.position + len_episode > self.memory_size:
            self.position = 0

        self.memory[self.position:self.position + len_episode] = episode_to_store

        self.position += len_episode
        self.len = min(self.memory_size, self.len + len_episode)

    def __len__(self):
        return self.len

class PrioritizedReplayMemory(AbstractReplay):
    def __init__(self, size, seed, alpha, beta, annealing_rate, eps=1e-6):
        # todo : include per
        raise NotImplementedError("Not available yet")
        self.transition = collections.namedtuple("Transition",
                                                 ["curr_state", "action", "reward", "next_state", "terminal",
                                                  "mission"])
        self.memory_size = int(size)
        self.memory = [None for _ in range(self.memory_size)]
        self.priorities = np.zeros(self.memory_size)

        self.position = 0
        # Parameters to modulate the amount of PER
        self.alpha = alpha
        self.beta = beta
        # Minimal probability
        self.eps = eps
        # Annealing beta
        self.annealing_rate = annealing_rate
        self.len = 0
        random.seed(seed)
        np.random.seed(seed)
        self.stored_transitions = []

    def add_transition(self, curr_state, action, reward, next_state, terminal, mission):
        self.memory[self.position] = \
            self.transition(curr_state=curr_state, action=action, reward=reward, next_state=next_state,
                            terminal=terminal, mission=mission)
        # Add the maximal priority
        if self.len == 0:
            self.priorities[self.position] = 1
        else:
            self.priorities[self.position] = self.priorities.max()

        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0

    def sample(self, batch_size):
        #normalized_priorities = np.power(self.priorities[:self.len], self.alpha) + self.eps
        #normalized_priorities /= normalized_priorities.sum()
        normalized_priorities = self.priorities[:self.len] / self.priorities[:self.len].sum()
        transition_idxs = np.random.choice(np.arange(self.len),
                                           size=batch_size, replace=False, p=normalized_priorities)
        self.beta = min(1, self.beta + self.annealing_rate)
        is_weights = np.power(self.len * self.priorities[transition_idxs], -self.beta)
        is_weights = is_weights / is_weights.max()
        op = operator.itemgetter(*transition_idxs)
        return op(self.memory), is_weights, transition_idxs

    def update(self, idxs, errors):
        errors = np.power(errors + self.eps, self.alpha)
        self.priorities[idxs] = errors

    def __len__(self):

        return self.len

    def store_transition(self, curr_state, action, reward, next_state, terminal, mission):
        self.stored_transitions.append(self.transition(curr_state, action, reward, next_state, terminal, mission))

    def add_hindsight_transitions(self, reward, mission, keep_last_transitions):
        # keep_last_transitions = 0 => keep the whole episode
        if keep_last_transitions == 0:
            keep = 0
        elif keep_last_transitions > 0:
            keep = max(len(self.stored_transitions) - keep_last_transitions, 0)
        # Update the last transition with hindsight replay
        self.memory[self.position] = self.stored_transitions[-1]._replace(reward=reward, mission=mission)
        # Update the position and the len of the memory size
        self.position += 1
        self.len = min(self.memory_size, self.len + 1)
        if self.position > self.memory_size - 1:
            self.position = 0
        # Update all the transitions of the current episode with hindsight replay
        for transition in self.stored_transitions[keep:-1]:
            self.memory[self.position] = transition._replace(mission=mission)
            # Update the position and the len of the memory size
            self.position += 1
            self.len = min(self.memory_size, self.len + 1)
            if self.position > self.memory_size - 1:
                self.position = 0

        self.erase_stored_transitions()

    def erase_stored_transitions(self):
        self.stored_transitions = []
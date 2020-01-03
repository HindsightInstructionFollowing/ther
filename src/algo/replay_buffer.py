import collections
import operator
import random
import numpy as np
import torch

from abc import ABC
from abc import abstractmethod

class AbstractReplay(ABC):
    def __init__(self, config):
        """
        All replay buffers are based on the same strategy, making hindsight and Prioritized Replay MUCH EASIER

        Sample are added to a temporary episode buffer
        When DONE is reach, the episode is added to the replay replay alongside a new one if HER is used
        """
        self.transition = collections.namedtuple("Transition",
                                                 ["current_state", "action", "reward", "next_state", "terminal",
                                                  "mission", "mission_length"])
        self.stored_transitions = []
        self.current_episode = []
        self.hindsight_reward = config["hindsight_reward"]
        self.use_her = config["use_her"]

        self.memory_size = int(config["size"])
        self.memory = [None for _ in range(self.memory_size)]
        self.len = 0
        self.position = 0

    def sample(self, batch_size):
        return random.sample(self.memory[:self.len], batch_size)

    def _store_episode(self, episode_to_store):
        len_episode = len(episode_to_store)
        # If episode is too long, start cycling through buffer again
        if self.position + len_episode > self.memory_size:
            self.position = 0
            assert len_episode < self.memory_size,\
                "Problem, buffer not large enough, memory_size {}, len_episode {}".format(self.memory_size, len_episode)
        # If new episode makes the buffer a bit longer, new length is defined
        elif self.position + len_episode >= self.len:
            self.len = self.position + len_episode

        self.memory[self.position:self.position + len_episode] = episode_to_store
        self.position += len_episode

    def update_transitions_proba(self):
        pass
    def __len__(self):
        return self.len

    @abstractmethod
    def add_transition(self, current_state, action, reward, next_state, terminal, mission, mission_length,
                       hindsight_mission):
        pass  # todo : remove mission_length, can be computed on the fly instead of moving it around

class ReplayMemory(AbstractReplay):
    def __init__(self, config):
        super().__init__(config)

    def add_transition(self, current_state, action, reward, next_state, terminal, mission, mission_length, hindsight_mission=None):
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
            self.transition(current_state, action, reward, next_state, terminal, mission, mission_length)
        )

        if hindsight_mission and self.use_her:
            assert terminal is True, "If hindsight mission is provided, should be at the end of episode, terminal == False"
            assert reward <= 0, "Hindsight mission should be provided only if objective failed. Reward : {}".format(reward)

            # Substitute the old mission with the new one, change the reward too
            hindsight_episode = []
            for st, a, wrong_reward, st_plus1, end_ep, wrong_mission, length in self.current_episode:
                hindsight_episode.append(self.transition(current_state=st,
                                                         action=a,
                                                         reward=self.hindsight_reward if end_ep else 0,
                                                         next_state=st_plus1,
                                                         terminal=end_ep,
                                                         mission=torch.LongTensor(hindsight_mission),
                                                         mission_length=torch.LongTensor([len(hindsight_mission)])))
            self._store_episode(hindsight_episode)

        if terminal:
            self._store_episode(self.current_episode)
            self.current_episode = []

class RecurrentReplayBuffer(ReplayMemory):
    def __init__(self, config):
        super().__init__(config=config)

        self.episode_length = np.zeros(self.memory_size)
        self.transition_proba = np.zeros(self.memory_size)

    def _store_episode(self, episode_to_store):
        """
        Recurrent buffer stores episode by episode instead of a flat representation
        This allows to retrieve entire episode, it's smoother to read and pad/batch

        In a flat representation, all samples are uniformely sampled
        Since the buffer sample episodes by episodes, we will replay samples in small episodes more frequently

        It could be cool, be we don't want to bias the sampling towards smaller sequences
        Hence, the transition_proba compensate for this bias
        """
        len_episode = len(episode_to_store)

        # If end of buffer is reach start cycling through buffer again
        if self.position + 1 > self.memory_size:
            self.position = 0

        # Store episode and length
        self.memory[self.position] = episode_to_store
        self.episode_length[self.position] = len_episode

        # Push selector and update buffer length if necessary
        self.position += 1
        self.len = max(self.len, self.position)

        # Update transition proba
        self.transition_proba[:self.len] = self.episode_length[:self.len] / self.episode_length[:self.len].sum()

    def sample(self, batch_size):
        return np.random.choice(self.memory[:self.len], batch_size, p=self.transition_proba[:self.len])



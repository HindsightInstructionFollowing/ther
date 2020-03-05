import operator
import random
import numpy as np
import torch

from abc import ABC
from abc import abstractmethod

from algo.compression import DummyCompressor, TransitionCompressor
from algo.neural_architecture import basic_transition

class AbstractReplay(ABC):
    def __init__(self, config):
        """
        All replay buffers are based on the same strategy, making hindsight and Prioritized Replay MUCH EASIER

        Sample are added to a temporary episode buffer
        When DONE is reach, the episode is added to the replay replay alongside a new one if HER is used
        """
        self.transition = basic_transition
        self.stored_transitions = []
        self.current_episode = []
        self.hindsight_reward = config["hindsight_reward"]
        self.use_her = config["use_her"]

        self.memory_size = int(config["size"])
        self.memory = [None for _ in range(self.memory_size)]

        #  1 memory cell = 1 transition for normal buffer OR 1 episode for Recurrent Buffer
        self.n_memory_cell = 0
        self.position = 0

        # Prioritized parameters
        self.use_prioritization =       config["prioritize"]
        self.prioritize_alpha = config["prioritize_alpha"]
        self.prioritize_beta =  config["prioritize_beta"]
        self.prioritize_eps   = config["prioritize_eps"]
        self.prioritize_p =     np.zeros(self.memory_size)
        self.id_range =         np.arange(self.memory_size)
        self.max_proba = 1
        self.prioritize_p[0] = 1

        # N-step, gamma
        self.last_terminal = []  # Logging done or terminal (end of episode)
        self.last_returns = []  # Logging rewards (return)
        self.last_states = []  # Logging next states
        self.last_transitions = []  # Logging rest of the transitions (current_state, action etc ...)
        self.n_step =      config["n_step"]

        self.gamma =       config["gamma"]
        self.gamma_array = np.array([self.gamma**i for i in range(self.n_step+1)])

        # Storing in replay buffer might be expensive, compressing states (images) can be an option
        # todo : compress in gpu, would be awesome
        if config["use_compression"]:
            self.compressor = TransitionCompressor()
        else:
            self.compressor = DummyCompressor()

    def sample(self, batch_size):

        if self.use_prioritization:
            prioritize_proba = self.prioritize_p / self.prioritize_p.sum()
            min_proba = np.min(prioritize_proba[:len(self)])

            self.last_id_sampled = np.random.choice(self.id_range,
                                                    size=batch_size,
                                                    replace=True,
                                                    p=prioritize_proba)

            max_w = (len(self) * min_proba) ** - self.prioritize_beta

            is_weights = np.power(prioritize_proba[self.last_id_sampled] * len(self), - self.prioritize_beta)
            is_weights /= max_w

        else:
            self.last_id_sampled = np.random.randint(0, self.n_memory_cell, batch_size)
            is_weights = np.ones(batch_size)

        return [self.compressor.decompress_transition(self.memory[id_sample])
                for id_sample in self.last_id_sampled], is_weights

    def _store_episode(self, episode_to_store):
        len_episode = len(episode_to_store)
        # If episode is too long, start cycling through buffer again
        if self.position + len_episode > self.memory_size:
            self.position = 0
            assert len_episode < self.memory_size,\
                "Problem, buffer not large enough, memory_size {}, len_episode {}".format(self.memory_size, len_episode)
        # If new episode makes the buffer a bit longer, new length is defined
        elif self.position + len_episode >= self.n_memory_cell:
            self.n_memory_cell = self.position + len_episode

        self.memory[self.position:self.position + len_episode] = episode_to_store
        max_prio = self.prioritize_p.max()
        self.prioritize_p[self.position:self.position + len_episode] = max_prio

        self.position += len_episode

    def add_n_step_transition(self, n_step):
        current_state, action, mission, mission_length = self.last_transitions[0]

        # Not enough samples to apply full n-step, aborting this step
        if len(self.last_returns) < n_step:
            return

        sum_return = np.sum(self.last_returns[:n_step] * self.gamma_array[:n_step])
        n_step_state = self.last_states[n_step - 1]
        last_terminal = self.last_terminal[n_step - 1]

        if not last_terminal:
            assert sum_return == 0

        self.current_episode.append(
            self.compressor.compress_transition(
                self.transition(current_state=current_state, action=action, reward=sum_return, next_state=n_step_state,
                                terminal=last_terminal, mission=mission, mission_length=mission_length,
                                gamma=self.gamma_array[n_step])
            )
        )

        # Clean
        self.last_terminal.pop(0)
        self.last_returns.pop(0)
        self.last_states.pop(0)
        self.last_transitions.pop(0)

    def update_transitions_proba(self, errors):
        delta = errors + self.prioritize_eps
        self.prioritize_p[self.last_id_sampled] = np.power(delta, self.prioritize_alpha)

    def __len__(self):
        return self.n_memory_cell

    @abstractmethod
    def add_transition(self, current_state, action, reward, next_state, terminal, mission, mission_length,
                       hindsight_mission, correct_obj_name):
        pass  # todo : remove mission_length, can be computed on the fly instead of moving it around

class ReplayMemory(AbstractReplay):
    def __init__(self, config):
        super().__init__(config)

    def add_transition(self, current_state, action, reward, next_state, terminal, mission, mission_length, hindsight_mission=None, correct_obj_name=None):
        """
        Adds transition to an temporary episode buffer
        When TERMINAL is reach, the episode is added to the replay buffer

        This allows to switch mission and reward for the whole episode, if necessary.

        If hindsight mission is provided (terminal == True is needed)
        The current episode is stored alongside a new one with the substitued mission

        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        HER works only if the reward is sparse and provided at the end of the episode
        because of nstep
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """

        # ============= Apply n-step and store transitions in temporary episode ====================
        # ==========================================================================================

        # Saving everything except reward and next_state
        self.last_transitions.append((current_state, action, mission, mission_length))
        self.last_terminal.append(terminal)
        self.last_returns.append(reward)
        self.last_states.append(next_state)

        # Implementation of n-step return
        # The main idea is (almost-)invisible for the main algorithm, we just change the reward and the next state
        # The reward is the cumulative return until n-1 and next_state is the state after n step
        if len(self.last_states) >= self.n_step:
            self.add_n_step_transition(self.n_step)

        # If terminal, add all sample even if n_step is not available
        if terminal:
            truncate_n_step = 1
            while len(self.last_transitions) > 0:
                current_n_step = self.n_step - truncate_n_step
                self.add_n_step_transition(n_step = current_n_step)
                truncate_n_step += 1

            assert len(self.last_terminal) == 0
            assert len(self.last_returns) == 0
            assert len(self.last_transitions) == 0
            self.last_states = []

        # ============= Apply Hinsight Experience Replay by swapping a mission =====================
        # ==========================================================================================
        if hindsight_mission and self.use_her:
            assert terminal is True, "If hindsight mission is provided, should be at the end of episode, terminal == False"
            assert reward <= 0, "Hindsight mission should be provided only if objective failed. Reward : {}".format(reward)

            # Substitute the old mission with the new one, change the reward too
            hindsight_episode = []
            len_episode = len(self.current_episode)
            for step, (st, a, wrong_reward, st_plus1, end_ep, wrong_mission, length, gamma) in enumerate(self.current_episode):
                if step >= len_episode - self.n_step:
                    hindsight_reward = 1 * self.gamma**(len(self.current_episode) - step - 1)
                    assert end_ep
                else:
                    hindsight_reward = 0


                hindsight_episode.append(self.transition(current_state=st,
                                                         action=a,
                                                         reward=hindsight_reward,
                                                         next_state=st_plus1,
                                                         terminal=end_ep,
                                                         mission=torch.LongTensor(hindsight_mission),
                                                         mission_length=torch.LongTensor([len(hindsight_mission)]),
                                                         gamma=gamma))
            self._store_episode(hindsight_episode)

        if terminal:
            self._store_episode(self.current_episode)
            self.current_episode = []

class RecurrentReplayBuffer(ReplayMemory):
    def __init__(self, config):

        self.len_sum = 0
        super().__init__(config=config)

        self.episode_length = np.zeros(self.memory_size)
        self.transition_proba = np.zeros(self.memory_size)
        self.last_position = 0

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
        if self.len_sum >= self.memory_size and self.position + 1 > self.last_position:
            self.last_position = self.position
            self.position = 0

        # Store episode and length
        self.memory[self.position] = episode_to_store
        self.episode_length[self.position] = len_episode

        # Push selector and update buffer length if necessary
        self.position += 1
        self.n_memory_cell = max(self.n_memory_cell, self.position)

        # Update transition proba
        self.transition_proba[:self.n_memory_cell] = self.episode_length[:self.n_memory_cell] / self.episode_length[:self.n_memory_cell].sum()
        self.len_sum = self.episode_length.sum()

    def sample(self, batch_size):

        size = 0
        batch_seq = []

        id_taken = set()

        while size < batch_size:
            id_episode = np.random.choice(range(self.n_memory_cell), p=self.transition_proba[:self.n_memory_cell])
            if id_episode in id_taken : continue
            seq = [self.compressor.decompress_transition(transition) for transition in self.memory[id_episode]]
            size += len(seq)
            batch_seq.append(seq)
            id_taken.add(id_episode)

        return batch_seq

    def __len__(self):
        return int(self.len_sum)


if __name__ == "__main__":
    config = {"hindsight_reward": 1,
              "size": 11,
              "use_her": True,
              "n_step": 4,
              "gamma" : 0.99,
              "use_compression" : False,

              "prioritize": True,
              "prioritize_alpha" : 0.9,
              "prioritize_beta": 0.6,
              "prioritize_eps" : 1e-6,
              }

    # %%

    buffer = RecurrentReplayBuffer(config)

    mission = ["nik tout"]
    action = 2
    reward = 0
    for elem in range(1, 20):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 5 == 0:
            print("done")
            done = True
            hindsight_mission = [1, 0]
            reward = 0

        buffer.add_transition(elem, action, reward, elem + 1, done, mission, 2, hindsight_mission)
        print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
            len(buffer), buffer.position, len(buffer.current_episode), elem))
        print("buffer memory", buffer.memory)
# encoding: utf-8

import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler

from algo.compression import BloscArrayCompressor

class MissionPadder(object):
    def __init__(self, max_length, padding_token):
        self.max_length = max_length
        self.padding_token = padding_token

    def __call__(self, mission):
        padding_length = self.max_length - len(mission)
        if padding_length > 0:
            mission = np.concatenate([mission, [self.padding_token] * padding_length])
        return mission

class WeightedSampler(Sampler):
    def __init__(self, memory_allocated, batch_size, prioritize, prioritize_beta=None):
        """
        :param memory_allocated: For recurrent, you need less memory slot (because a slot contains an episode) so
        memory_allocated can be smaller than the number of transitions the replay buffer contains

        :param batch_size: Fixed in advance
        :param prioritize:
        :param prioritize_beta:
        """
        super().__init__(0)

        self.batch_size = batch_size
        self.use_prioritization = prioritize

        self.prioritize_p = np.zeros(memory_allocated)
        self.id_range = np.arange(memory_allocated)

        self.position = 0
        self.n_memory_cell = 0

        self.max_proba = 1
        self.prioritize_p[0] = 1

        if self.use_prioritization:
            self.prioritize_beta = prioritize_beta

    def add_transition_proba(self, old_position, new_position, new_n_memory_cell):
        max_prio = self.prioritize_p.max()
        self.prioritize_p[old_position:new_position] = max_prio

        self.n_memory_cell = new_n_memory_cell

    def update_transitions_proba(self, errors, id_retrieved):
        self.prioritize_p[id_retrieved] = errors

    def compute_is_weights(self, id_retrieved):
        prioritize_proba = self.prioritize_p[:self.n_memory_cell] / self.prioritize_p[:self.n_memory_cell].sum()
        min_proba = np.min(prioritize_proba)
        max_w = (len(self) * min_proba) ** - self.prioritize_beta
        is_weights = np.power(prioritize_proba[id_retrieved] * len(self), - self.prioritize_beta)
        is_weights /= max_w

        return is_weights

    def __len__(self):
        return self.n_memory_cell # Number of cell used the replay buffer

    def __iter__(self):
        n_max_sampling = self.n_memory_cell
        if self.use_prioritization:
            while True:
                prioritize_proba = self.prioritize_p[:n_max_sampling] / self.prioritize_p[:n_max_sampling].sum()
                yield np.random.choice(self.id_range[:n_max_sampling],
                                       size=self.batch_size,
                                       p=prioritize_proba[:n_max_sampling])
        else:
            while True:
                yield np.random.choice(self.id_range[:n_max_sampling],
                                       size=self.batch_size)

class MemoryDataset(Dataset):
    def __init__(self, memory_size, compressor,
                 state_padding=None, action_padding=None, terminal_padding=None, max_length_seq=None, memory_allocated=None):

        self.memory_size = memory_size
        self.memory = [None for _ in range(self.memory_size)]

        self.compressor = compressor

        #  1 memory cell = 1 transition for normal buffer OR 1 episode for Recurrent Buffer
        self.n_memory_cell = 0
        self.position = 0

    def store_episode(self, episode_to_store):
        len_episode = len(episode_to_store)
        # If episode is too long, start cycling through buffer again
        if self.position + len_episode > self.memory_size:
            self.position = 0
            assert len_episode < self.memory_size, \
                "Problem, buffer not large enough, memory_size {}, len_episode {}".format(self.memory_size, len_episode)
        # If new episode makes the buffer a bit longer, new length is defined
        elif self.position + len_episode >= self.n_memory_cell:
            self.n_memory_cell = self.position + len_episode

        self.memory[self.position:self.position + len_episode] = episode_to_store
        old_position = self.position
        self.position += len_episode

        return old_position, self.position, self.n_memory_cell

    def __getitem__(self, id_transition):

        transition = self.memory[id_transition]

        batch_dict = {
            "state":          self.compressor.decompress_elem(transition.current_state),
            "next_state":     self.compressor.decompress_elem(transition.current_state),
            "mission":        self.mission_padder(transition.mission),
            "mission_length": transition.mission_length,
            "terminal":       transition.terminal,
            "action":         transition.action,
            "reward":         transition.reward,
            "gamma":          transition.gamma
        }

        return batch_dict

    def __len__(self):
        return self.n_memory_cell

class MemoryDatasetRecurrent(MemoryDataset):
    def __init__(self, memory_size, compressor, state_padding, action_padding, terminal_padding, max_length_seq, memory_allocated):

        super().__init__(memory_size, compressor)

        # Override to save some space
        self.memory = [None for _ in range(memory_allocated)]
        self.episode_length = np.zeros(memory_allocated)
        self.episode_identificator = np.zeros(memory_allocated)

        self.id_current_episode = 0

        self.max_length_seq = max_length_seq
        self.len_sum = 0
        self.last_position = 0

        self.compressor = compressor
        self.state_padding = state_padding
        self.action_padding = action_padding
        self.terminal_padding = terminal_padding

    def store_episode(self, episode_to_store):
        """
        Recurrent buffer stores episode by episode instead of a flat representation
        This allows to retrieve entire episode, it's smoother to read and pad/batch

        In a flat representation, all samples are uniformely sampled
        Since the buffer samples episodes by episodes, we will replay samples in small episodes more frequently

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
        self.episode_identificator[self.position] = self.id_current_episode
        self.id_current_episode += 1

        # Push selector and update buffer length if necessary
        self.position += 1
        self.n_memory_cell = max(self.n_memory_cell, self.position)

        self.len_sum = self.episode_length.sum()
        return self.position - 1, self.position, self.n_memory_cell

    def __len__(self):
        return len(self.len_sum)

    def __getitem__(self, id_episode):

        batch_dict = {
            "state": [],
            "next_state": [],
            "mission": [],
            "mission_length": [],
            "terminal": [],
            "last_action": [],
            "action": [],
            "reward": [],
            "gamma": [],
            "id_retrieved" : [id_episode],
            "id_verification" : [self.episode_identificator[id_episode]]
        }

        sequence = self.memory[id_episode]
        #print(id_episode, self.n_memory_cell)
        len_sequence = len(sequence)
        assert self.episode_length[id_episode] == len_sequence

        for transition in sequence:

            batch_dict["state"].append(         self.compressor.decompress_elem(transition.current_state))
            batch_dict["next_state"].append(    self.compressor.decompress_elem(transition.next_state))
            batch_dict["mission_length"].append(transition.mission_length)
            batch_dict["terminal"].append(      int(transition.terminal))
            batch_dict["action"].append(        transition.action)
            batch_dict["mission"].append(       transition.mission)
            batch_dict["reward"].append(        transition.reward)
            batch_dict["gamma"].append(         transition.gamma)

        padding_length = self.max_length_seq - len_sequence
        if padding_length > 0:

            batch_dict["state"] +=          [self.state_padding] * padding_length
            batch_dict["next_state"] +=     [self.state_padding] * padding_length

            batch_dict["terminal"] +=       [self.terminal_padding] * padding_length
            batch_dict["action"] +=         [self.action_padding] * padding_length

            batch_dict["mission"] +=        [batch_dict["mission"][0]] * padding_length
            batch_dict["mission_length"] += [batch_dict["mission_length"][0]] * padding_length
            batch_dict["reward"] +=         [0] * padding_length
            batch_dict["gamma"] +=          [0] * padding_length


        batch_dict["last_action"].append(0) # Doesn't exist for the 1st step, so anything is fine
        batch_dict["last_action"] += batch_dict["action"][:-1]

        batch_dict["action"] =      np.array(batch_dict["action"])
        batch_dict["last_action"] = np.array(batch_dict["last_action"])
        batch_dict["gamma"] =       np.array(batch_dict["gamma"])
        batch_dict["terminal"] =    np.array(batch_dict["terminal"])
        batch_dict["reward"] =      np.array(batch_dict["reward"])

        for key in ["state", "next_state", "mission", "mission_length"]:
            batch_dict[key] = np.concatenate(batch_dict[key])

        mask = np.ones(self.max_length_seq)
        mask[len_sequence:] = 0
        batch_dict["padding_mask"] = mask
        batch_dict["state_sequence_lengths"] = np.array([len_sequence])

        batch_dict["mission_length"] = batch_dict["mission_length"].reshape(-1)

        return batch_dict



if __name__ == "__main__":

    import dill
    import torch
    import time

    full_buffer = dill.load(open("saved_tools/buffer_filled", 'rb'))
    compressor = BloscArrayCompressor()
    mission_padder = MissionPadder(8, 0)

    state_padding = torch.zeros(1, 3, 84, 84)
    action_padding = 4
    terminal_padding = 1

    dataset = MemoryDatasetRecurrent(memory=full_buffer.memory, compressor=compressor,
                                     state_padding=state_padding, action_padding=action_padding,
                                     terminal_padding=terminal_padding, max_length_seq=30, mission_padder=mission_padder)

    custom_sampler = WeightedSampler(batch_size=24, prioritize=True)

    dataloader = DataLoader(dataset=dataset,
                            num_workers=4,
                            batch_sampler=custom_sampler)



    a = time.time()
    for i in dataloader:
        b = time.time()
        print(b - a)
        a = b


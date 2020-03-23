# encoding: utf-8

import numpy as np
import torch
from torch.utils.data import DataLoader

from abc import ABC
from abc import abstractmethod

from algo.compression import DummyCompressor, BloscArrayCompressor
from algo.transition import basic_transition

from algo.replay_buffer_tools import WeightedSampler, MemoryDatasetRecurrent, MemoryDataset



from torch._six import container_abcs, string_classes, int_classes
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}
import re
_use_shared_memory = False
np_str_obj_array_pattern = re.compile(r'[SaUO]')

error_msg_fmt = "batch must contain tensors, numbers, dicts or lists; found {}"
def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)

        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(error_msg_fmt.format(elem.dtype))

            return default_collate([torch.from_numpy(b) for b in batch])
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(batch[0], int_classes):
        return torch.tensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], tuple) and hasattr(batch[0], '_fields'):  # namedtuple
        return type(batch[0])(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(batch[0], container_abcs.Sequence):
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError((error_msg_fmt.format(type(batch[0]))))

class ReplayBufferParallel(DataLoader):
    def __init__(self, config, is_recurrent, env=None, recurrent_memory_saving=10, logger=None, device=None):
        """
        All replay buffers are based on Dataloader, to allow parallel retrieving and preformating
        """
        self.stored_transitions = []
        self.current_episode = []
        self.hindsight_reward = config["hindsight_reward"]
        self.use_her = config["use_her"]
        self.is_recurrent = is_recurrent

        # ====================== COMPRESSOR ====================
        # Storing in replay buffer might be expensive, compressing states (images) can be an option
        if config["use_compression"]:
            self.compressor = BloscArrayCompressor()
        else:
            self.compressor = DummyCompressor()

        # ====================== DATASET CREATION ====================
        memory_size = int(config["size"])

        if is_recurrent:
            dataset_creator = MemoryDatasetRecurrent
            memory_allocated = memory_size // recurrent_memory_saving # Don't allocate too much cell for recurrent
        else:
            dataset_creator = MemoryDataset
            memory_allocated = memory_size

        dataset = dataset_creator(memory_size=memory_size,
                                  compressor=self.compressor,
                                  state_padding=np.zeros((1, *env.observation_space["image"].shape)).astype(np.float32),
                                  action_padding=env.action_space.n,
                                  terminal_padding=1,
                                  max_length_seq=env.max_steps,
                                  memory_allocated=memory_allocated
                                  )

        # ==================== SAMPLER / PRIORITIZER ================
        self.use_prioritization = config["prioritize"]
        if self.use_prioritization:
            self.prioritize_alpha = config["prioritize_alpha"],
            self.prioritize_eps = config["prioritize_eps"],
            self.prioritize_max_mean_balance = config["prioritize_max_mean_balance"]

        weighted_sampler = WeightedSampler(memory_allocated=memory_allocated,
                                           batch_size=config["batch_size"],
                                           prioritize=self.use_prioritization,
                                           prioritize_beta = config["prioritize_beta"]
                                           )
        # ==================== N STEP and HORIZON ====================
        # N-step, gamma
        self.last_terminal = []  # Logging done or terminal (end of episode)
        self.last_returns = []  # Logging rewards (return)
        self.last_states = []  # Logging next states
        self.last_transitions = []  # Logging rest of the transitions (current_state, action etc ...)

        self.n_step = config["n_step"]
        self.gamma =  config["gamma"]

        self.gamma_array = np.array([self.gamma**i for i in range(self.n_step+1)])

        # ====================  LOADER ITERATOR  ======================
        super().__init__(dataset=dataset,
                         num_workers=config["num_workers"],
                         batch_sampler=weighted_sampler,
                         collate_fn=default_collate,
                         pin_memory=True)

        self.iter = None

    def sample(self):
        if self.iter is None:
            self.iter = self.__iter__()
        self.batch_sampler.compute_prioritize_proba()
        batch = next(self.iter)
        return batch, self.compute_is_weights(batch["id_retrieved"][0])

    def _store_episode(self, episode_to_store):
        old_position, new_position, new_n_memory_cell = self.dataset.store_episode(episode_to_store)
        self.batch_sampler.add_transition_proba(old_position=old_position,
                                                new_position=new_position,
                                                new_n_memory_cell=new_n_memory_cell)

        assert self.batch_sampler.n_memory_cell == self.dataset.n_memory_cell

    def update_transitions_proba(self, errors, id_retrieved):
        """
        Two Cases needed here because the way the error are computed between recurrent and normal replay buffer

        Recurrent :
            The strategy employed in R2D2 : https://openreview.net/pdf?id=r1lyTjAqYX

            The p of a sequence  is : η * max_i δi + (1−η) * mean(δ)
            with   δ = td-err ** alpha

            η is self.prioritize_max_mean_balance, balancing between sequence's max and mean TD error

        Not recurrent :
            p = (td-error + epsilon) ** alpha to avoid priority 0
        """
        if self.use_prioritization:
            if self.is_recurrent:
                processed_errors = []
                last_l = 0
                errors = np.power(errors, self.prioritize_alpha)
                for l in self.dataset.episode_length[id_retrieved]:
                    l = int(l)
                    err_list = errors[last_l:last_l + l]
                    err = self.prioritize_max_mean_balance * np.max(err_list) + (
                                1 - self.prioritize_max_mean_balance) * np.mean(err_list)
                    processed_errors.append(err)
                    last_l += l
            else:
                delta = errors + self.prioritize_eps
                processed_errors = np.power(delta, self.prioritize_alpha)

            self.batch_sampler.update_transitions_proba(processed_errors, id_retrieved)

    def compute_is_weights(self, id_retrieved):
        if self.use_prioritization:
            is_weights = self.batch_sampler.compute_is_weights(id_retrieved)
            if self.is_recurrent:
                expand_is_weight_list = []
                for num_weight, length in enumerate(self.dataset.episode_length[id_retrieved]):
                    temp_w = np.empty(int(length))
                    temp_w[:] = is_weights[num_weight]
                    expand_is_weight_list.append(temp_w)

                is_weights = np.concatenate(expand_is_weight_list)
            return is_weights
        return 1

    def add_transition(self, current_state, action, reward, next_state, terminal, mission, mission_length,
                       hindsight_mission=None, correct_obj_name=None):
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
            self.add_n_step_transition_to_ep(self.n_step)

        # If terminal, add all sample even if n_step is not available
        if terminal:
            truncate_n_step = 1
            while len(self.last_transitions) > 0:
                current_n_step = self.n_step - truncate_n_step
                self.add_n_step_transition_to_ep(n_step=current_n_step)
                truncate_n_step += 1

            assert len(self.last_terminal) == 0
            assert len(self.last_returns) == 0
            assert len(self.last_transitions) == 0
            self.last_states = []

        # ============= Apply Hinsight Experience Replay by swapping a mission =====================
        # ==========================================================================================
        if hindsight_mission and self.use_her:
            assert terminal is True, "If hindsight mission is provided, should be at the end of episode, terminal == False"
            assert reward <= 0, "Hindsight mission should be provided only if objective failed. Reward : {}".format(
                reward)

            # Substitute the old mission with the new one, change the reward too
            hindsight_episode = []
            len_episode = len(self.current_episode)
            for step, (st, a, wrong_reward, st_plus1, end_ep, wrong_mission, length, gamma) in enumerate(
                    self.current_episode):
                if step >= len_episode - self.n_step:
                    hindsight_reward = 1 * self.gamma ** (len(self.current_episode) - step - 1)
                    assert end_ep
                else:
                    hindsight_reward = 0

                hindsight_episode.append(basic_transition(current_state=st,
                                                          action=a,
                                                          reward=hindsight_reward,
                                                          next_state=st_plus1,
                                                          terminal=end_ep,
                                                          mission=hindsight_mission,
                                                          mission_length=np.array([len(hindsight_mission)]),
                                                          gamma=gamma))
            self._store_episode(hindsight_episode)

        if terminal:
            self._store_episode(self.current_episode)
            self.current_episode = []

    def add_n_step_transition_to_ep(self, n_step):
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
            basic_transition(
                current_state=self.compressor.compress_elem(current_state),
                next_state=self.compressor.compress_elem(n_step_state),
                action=action,
                reward=sum_return,
                terminal=last_terminal,
                mission=mission,
                mission_length=np.array([mission_length]),
                gamma=self.gamma_array[n_step]
            )
        )

        # Clean
        self.last_terminal.pop(0)
        self.last_returns.pop(0)
        self.last_states.pop(0)
        self.last_transitions.pop(0)

    def __len__(self):
        return self.dataset.n_memory_cell

    def generate(self, *args, **kwargs):
        raise AttributeError("generate() is not available in generic replay buffer")

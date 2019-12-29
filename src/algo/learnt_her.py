from algo.replay_buffer import AbstractReplay
from algo.neural_architecture import InstructionGenerator
import numpy as np
import torch
import torch.nn.functional as F

class LearntHindsightExperienceReplay(AbstractReplay):
    def __init__(self, input_shape, n_output, config, device, logger):
        super().__init__(config)

        config = config["ther_params"]

        self.instruction_generator = InstructionGenerator(input_shape=input_shape,
                                                          n_output=n_output,
                                                          config=config["architecture_params"],
                                                          device=device)

        # Self useful variable
        self.update_steps =                    set(config["update_steps"])
        self.n_sample_before_using_generator = config["n_sample_before_using_generator"]
        self.batch_size =                      config["batch_size"]
        self.loss =                            F.cross_entropy

        # Init generator dataset, device and logger
        self.device = device
        self.generator_dataset = {'states': [], 'instructions': [], 'lengths': []}
        self.instruction_generator.to(self.device)
        self.logger = logger

    def add_transition(self, current_state, action, reward, next_state, terminal, mission, mission_length, hindsight_mission=None):
        self.current_episode.append(
            self.transition(current_state, action, reward, next_state, terminal, mission, mission.size(0))
        )

        # Number of sample in the generator dataset
        n_generator_example = len(self.generator_dataset['states'])
        # If there are enough training example, train the generator
        if n_generator_example in self.update_steps:
            self._train_generator()

        if terminal:
            # Agent failed (reward <= 0), use the generator to compute the instruction performed by the agent
            if reward <= 0 and n_generator_example > self.n_sample_before_using_generator:
                assert hindsight_mission is not None, \
                    "Environment didn't provide hindsight mission, weird ! (Even though it's not used here)"

                last_transition = self.current_episode[-1]
                last_state =      last_transition.current_state

                # todo : in the long run, should take the whole trajectory as input
                hindsight_mission = self.instruction_generator.generate(last_state) # check last state represents what you want

                # Substitute the old mission with the new one, change the reward too
                hindsight_episode = [self.transition(st, a, self.hindsight_reward, st_plus1, end_ep, hindsight_mission, len(hindsight_mission))
                                                 for st, a, wrong_reward,          st_plus1, end_ep, wrong_mission,     length in self.current_episode]

                self._store_episode(hindsight_episode)

            # If the agent succeeded, store the state/instruction pair to train the generator
            elif reward > 0:
                self.generator_dataset["states"].append(current_state)
                self.generator_dataset["instructions"].append(mission)
                self.generator_dataset["lengths"].append(mission.size(0))

            self._store_episode(self.current_episode)
            self.current_episode = []

    def _train_generator(self):

        convergence = False
        len_dataset = len(self.generator_dataset["states"])
        # To speed batching, convert list to tensor
        states, lengths = torch.cat(self.generator_dataset['states']), torch.LongTensor(self.generator_dataset['lengths'])
        instructions = torch.nn.utils.rnn.pad_sequence(sequences=self.generator_dataset["instructions"],
                                                       batch_first=True,
                                                       padding_value=2  # Padding is always 2, checked by vocab
                                                       ) # check ordering is kept

        while not convergence:
            batch_idx = np.random.choice(range(len_dataset), self.batch_size)
            batch_state, batch_lengths = states[batch_idx].to(self.device), lengths[batch_idx].to(self.device)
            batch_instruction = instructions[batch_idx].to(self.device)

            softmaxes = self.instruction_generator(batch_state, batch_instruction, batch_lengths)


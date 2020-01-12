from algo.replay_buffer import AbstractReplay
from algo.neural_architecture import InstructionGenerator, compute_accuracy
import numpy as np
import torch
import torch.nn.functional as F

class LearntHindsightExperienceReplay(AbstractReplay):
    def __init__(self, input_shape, n_output, config, device, logger=None):
        super().__init__(config)

        config = config["ther_params"]

        # Force to take the last obs, not state
        input_shape = list(input_shape)
        input_shape[0] = 3

        self.instruction_generator = InstructionGenerator(input_shape=input_shape,
                                                          n_output=n_output,
                                                          config=config["architecture_params"],
                                                          device=device)

        # Self useful variable
        self.update_steps =                    config["update_steps"]
        self.n_sample_before_using_generator = config["n_sample_before_using_generator"]
        self.batch_size =                      config["batch_size"]
        self.accuracy_convergence =            config["accuracy_convergence"] # Accuracy before generator is considered good
        self.padding_value =                   2 # By convention, checked by Word2Idx in wrappers.py
        self.n_update_generator =              0 # Number of optim steps done on generator
        self.generator_usage =                 0 # Number of time a sequence has been relabeled
        self.generator_next_update =           self.update_steps

        # Loss and optimization
        self.loss =      F.cross_entropy
        self.optimizer = torch.optim.Adam(params=self.instruction_generator.parameters(),
                                          lr=config["lr"], weight_decay=config["weight_decay"])


        # Init generator dataset, device and logger
        self.device = device
        self.generator_dataset = {'states': [], 'instructions': [], 'lengths': []}
        self.instruction_generator.to(self.device)
        self.logger = logger

    def _cheat_check(self, true_mission, generated_mission):

        good_attributes = 0
        for i in range(1,5):
            if true_mission[-i] in generated_mission:
                good_attributes += 1

        return good_attributes

    def add_transition(self, current_state, action, reward, next_state, terminal, mission, mission_length, hindsight_mission):

        # ============= Apply n-step and store transitions in temporary episode ====================
        # ==========================================================================================

        # Saving everything except reward and next_state
        self.last_transitions.append((current_state, action, terminal, mission, mission_length))
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
                    self.add_n_step_transition(n_step=current_n_step)
                    truncate_n_step += 1

                assert len(self.last_returns) == 0
                assert len(self.last_transitions) == 0
                self.last_states = []

        # ==================== Deal with generator training and dataset ============================
        # ==========================================================================================
        self.logger.log("gen/len_dataset", len(self.generator_dataset["states"]))
        # Number of sample in the generator dataset
        n_generator_example = len(self.generator_dataset['states'])
        # If there are enough training example, train the generator
        if n_generator_example > self.generator_next_update:
            self._train_generator()
            self.generator_next_update += self.update_steps

        # ============= Apply Hinsight Experience Replay by swapping a mission =====================
        # ==========================================================================================
        if terminal:
            # Agent failed (reward <= 0), use the generator to compute the instruction performed by the agent
            if hindsight_mission and n_generator_example > self.n_sample_before_using_generator:

                last_transition = self.current_episode[-1]
                last_state =      last_transition.current_state

                # todo : in the long run, should take the whole trajectory as input
                last_state = last_state[:, -3:].to(self.device)
                generated_hindsight_mission = self.instruction_generator.generate(last_state)

                if last_state.size(2) == 7:
                    n_correct_attrib = self._cheat_check(true_mission=hindsight_mission,
                                                         generated_mission=generated_hindsight_mission)

                    self.logger.log("gen/n_correct_attrib_gen", n_correct_attrib, self.generator_usage)
                    self.generator_usage += 1

                # Substitute the old mission with the new one, change the reward at the end of episode
                hindsight_episode = []
                for st, a, wrong_reward, st_plus1, end_ep, wrong_mission, length in self.current_episode:
                    hindsight_reward = self.hindsight_reward if end_ep else 0
                    len_mission = torch.LongTensor([len(generated_hindsight_mission)])
                    hindsight_episode.append(
                        self.transition(st, a, hindsight_reward, st_plus1, end_ep, generated_hindsight_mission, len_mission)
                    )
                self._store_episode(hindsight_episode)

            # If the agent succeeded, store the state/instruction pair to train the generator
            elif reward > 0:
                self.generator_dataset["states"].append(current_state[:,-3:])
                self.generator_dataset["instructions"].append(mission)
                self.generator_dataset["lengths"].append(mission.size(0))

            self._store_episode(self.current_episode)
            self.current_episode = []

    def _train_generator(self):

        convergence = False
        accuracies = []
        len_dataset = len(self.generator_dataset["states"])
        # To speed batching, convert list to tensor
        states = torch.cat(self.generator_dataset['states'])
        lengths = self.generator_dataset['lengths']
        instructions = self.generator_dataset["instructions"]
        instructions, lengths = zip(*sorted(zip(instructions, lengths),
                                            key=lambda x: -x[0].size(0)))

        lengths = torch.LongTensor(lengths)
        instructions = torch.nn.utils.rnn.pad_sequence(sequences=instructions,
                                                       batch_first=True,
                                                       padding_value=self.padding_value
                                                       )

        while not convergence:
            batch_idx = np.random.choice(range(len_dataset), self.batch_size)
            batch_state, batch_lengths = states[batch_idx].to(self.device), lengths[batch_idx].to(self.device)
            batch_instruction = instructions[batch_idx].to(self.device)

            logits = self.instruction_generator(batch_state, batch_instruction, batch_lengths)

            # Turn instructions into labels for the generator
            instruction_label = batch_instruction[:,1:] # Remove <BEG> token

            # Pack padded sequence in .forward remove entire columns if last column is filled with <pad>
            # So we need to adjust the labels so the number of logits and label matches
            max_length = lengths.max().item()
            # If the last element of the longer sequence is <END> we need to add a <PAD> token.
            # Label size must matches output size
            if max_length in batch_lengths:
                instruction_label = torch.cat((instruction_label, torch.ones(self.batch_size, 1).fill_(self.padding_value).long().to(self.device)), dim=1)
            else:
                # If all sequences are filled with <PAD> at the end, they will not be computed by generator,
                # So we also remove them from labels
                current_max_length = batch_lengths.max().item()
                id_to_remove = max_length - current_max_length - 1
                if id_to_remove >= 1:
                    instruction_label = instruction_label[:,:-id_to_remove]

            instruction_label = instruction_label.reshape(-1)

            # Compute only index where no padding token is being predicted
            assert instruction_label.size(0) == logits.size(0), \
                "instruction {} logits {}, lengths {}".format(instruction_label.size(), logits.size(), batch_lengths)
            indexes = np.where(instruction_label.cpu() != self.padding_value)
            logits = logits[indexes]
            instruction_label = instruction_label[indexes]

            accuracy = compute_accuracy(logits, instruction_label)
            accuracies.append(accuracy)

            # Last round of test, check gradients and size
            assert logits.requires_grad is True
            assert instruction_label.requires_grad is False
            assert instruction_label.size(0) == logits.size(0)

            loss = self.loss(input=logits, target=instruction_label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.logger:
                self.logger.add_scalar("gen/generator_loss", loss.detach().item(), self.n_update_generator)
                self.logger.add_scalar("gen/generator_accuracy", accuracy, self.n_update_generator)

            self.n_update_generator += 1
            if np.mean(accuracies[-10:]) > self.accuracy_convergence:
                convergence = True

        print("Done training generator in {} steps\nLast accuracies : {}".format(self.n_update_generator, accuracies[-10:]))


if __name__ == "__main__":
    import pickle as pkl
    import random
    import json

    vocab = json.load(open("gym-minigrid/gym_minigrid/envs/missions/vocab_fetch.json", "r"))
    i2w = list(vocab["vocabulary"].keys())

    input_shape = (4,7,7)
    n_output = 27
    config = {
        "hindsight_reward" : 0.8,
        "use_her" : False,
        "size" : 40000,
        "ther_params": {
            "accuracy_convergence": 0.95,
            "lr": 3e-4,
            "batch_size": 4,
            "weight_decay": 0,
            "update_steps": [30, 300, 1000],
            "n_sample_before_using_generator": 300,

            "architecture_params": {
                "conv_layers_channel": [16, 32, 64],
                "conv_layers_size": [2, 2, 2],
                "conv_layers_stride": [1, 1, 1],
                "max_pool_layers": [2, 0, 0],
                "embedding_dim": 32,
                "generator_max_len": 10
            }
        }}

    replay = LearntHindsightExperienceReplay(input_shape, n_output, config, 'cpu')
    replay.generator_dataset = pkl.load(open("gen_dataset.pkl", "rb"))
    replay._train_generator()

    while True:
        state = random.choice(replay.generator_dataset["states"])
        instruction = replay.instruction_generator.generate(state)

        print(state)
        raw_instruction = []
        for word in instruction:
            raw_instruction.append(i2w[word])
        print(raw_instruction)
        input()
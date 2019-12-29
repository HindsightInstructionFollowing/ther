from algo.replay_buffer import AbstractReplay
from algo.neural_architecture import InstructionGenerator, compute_accuracy
import numpy as np
import torch
import torch.nn.functional as F

class LearntHindsightExperienceReplay(AbstractReplay):
    def __init__(self, input_shape, n_output, config, device, logger=None):
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
        self.padding_value =                   2 # By convention, checked by Word2Idx in wrappers.py

        # Loss and optimization
        self.loss =      F.cross_entropy
        self.optimizer = torch.optim.Adam(params=self.instruction_generator.parameters(),
                                          lr=config["lr"], weight_decay=config["weight_decay"])


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
        states = torch.cat(self.generator_dataset['states'])
        lengths = self.generator_dataset['lengths']
        instructions = self.generator_dataset["instructions"]
        instructions, lengths = zip(*sorted(zip(instructions, lengths),
                                            key=lambda x: -x[0].size(0)))

        lengths = torch.LongTensor(lengths)

        # for i in range(len(self.generator_dataset)):
        #     assert len(self.generator_dataset['instructions'][i]) == self.generator_dataset['lengths'][i]
        # print("Lengths are okay!")


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

            if lengths.max().item() in batch_lengths:
                instruction_label = torch.cat((instruction_label, torch.ones(self.batch_size, 1).fill_(self.padding_value).long()), dim=1)
            elif torch.all(batch_lengths == lengths.min().item()):
                instruction_label = instruction_label[:,:-1]

            instruction_label = instruction_label.reshape(-1)

            # Compute only index where no padding token is being predicted
            assert instruction_label.size(0) == logits.size(0), \
                "instruction {} logits {}, lengths {}".format(instruction_label.size(), logits.size(), lengths)
            indexes = np.where(instruction_label != self.padding_value)
            logits = logits[indexes]
            instruction_label = instruction_label[indexes]

            accuracy = compute_accuracy(logits, instruction_label)

            # Last round of test
            assert logits.requires_grad is True
            assert instruction_label.requires_grad is False
            assert instruction_label.size(0) == logits.size(0)

            loss = self.loss(input=logits, target=instruction_label)

            print(loss, accuracy)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


if __name__ == "__main__":
    import pickle as pkl

    input_shape = (4,7,7)
    n_output = 27
    config = {
        "hindsight_reward" : 0.8,
        "use_her" : False,
        "size" : 40000,
        "ther_params": {
            "lr": 3e-4,
            "batch_size": 5,
            "weight_decay": 0,
            "update_steps": [10, 300, 1000],
            "n_sample_before_using_generator": 300,

            "architecture_params": {
                "conv_layers_channel": [16, 32, 64],
                "conv_layers_size": [2, 2, 2],
                "conv_layers_stride": [1, 1, 1],
                "max_pool_layers": [2, 0, 0],
                "embedding_dim": 32
            }
        }}

    replay = LearntHindsightExperienceReplay(input_shape, n_output, config, 'cpu')
    replay.generator_dataset = pkl.load(open("gen_dataset.pkl", "rb"))
    replay._train_generator()
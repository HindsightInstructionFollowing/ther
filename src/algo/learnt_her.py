from algo.replay_buffer import AbstractReplay, RecurrentReplayBuffer
from algo.neural_architecture import InstructionGenerator, compute_accuracy
import numpy as np
import torch
import torch.nn.functional as F
import pickle as pkl

from nltk.translate import bleu_score
import time

class LearntHindsightExperienceReplay(AbstractReplay):
    def __init__(self, input_shape, n_output, config, device, logger=None):
        super().__init__(config)

        config = config["ther_params"]

        # Scheduling and usage
        self.update_steps                    = config["update_steps"]
        self.n_sample_before_using_generator = config["n_sample_before_using_generator"]

        # Convention and usage variable
        self.padding_value =                   2 # By convention, checked by Word2Idx in wrappers.py
        self.n_state_to_predict_instruction =  config["n_state_to_predict_instruction"]

        # Model mapping state(s) to an instruction
        self.dummy_state = torch.zeros(input_shape).unsqueeze(0)
        config["architecture_params"]["n_state_to_predict_instruction"] = self.n_state_to_predict_instruction
        self.instruction_generator = InstructionGenerator(input_shape=input_shape,
                                                          n_output=n_output,
                                                          config=config["architecture_params"],
                                                          device=device)

        # Loss and optimization
        self.batch_size           = config["batch_size"]
        self.max_steps_optim =      config["max_steps_optim"]
        self.accuracy_convergence = config["accuracy_convergence"]
        self.loss =                 F.cross_entropy
        self.optimizer =            torch.optim.Adam(params=self.instruction_generator.parameters(),
                                                     lr=config["lr"], weight_decay=config["weight_decay"])


        # Init generator dataset, device and logger
        self.device = device
        self.generator_dataset = {'states': [], 'instructions': [], 'lengths': [], 'correct_obj_name' : []}
        self.instruction_generator.to(self.device)
        self.logger = logger
        self.n_update_generator = 0  # Number of optim steps done on generator
        self.generator_usage = 0  # Number of time a sequence has been relabeled
        self.generator_next_update =           self.update_steps

    def _cheat_check(self, true_mission, generated_mission):

        good_attributes = 0
        for i in range(2,6):
            if true_mission[-i] in generated_mission:
                good_attributes += 1

        return good_attributes

    def add_transition(self, current_state, action, reward, next_state, terminal, mission, mission_length, hindsight_mission, correct_obj_name):

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
                self.add_n_step_transition(n_step=current_n_step)
                truncate_n_step += 1

            assert len(self.last_returns) == 0
            assert len(self.last_transitions) == 0
            assert len(self.last_terminal) == 0
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

                last_states =     self.current_episode[-self.n_state_to_predict_instruction:]
                trajectory =      [self.compressor.decompress_elem(t.current_state) for t in last_states]
                if len(trajectory) < self.n_state_to_predict_instruction:
                    trajectory =      [self.dummy_state] * (self.n_state_to_predict_instruction - len(trajectory)) + trajectory
                trajectory =      torch.cat(trajectory, dim=0).to(self.device)

                generated_hindsight_mission = self.instruction_generator.generate(trajectory)

                if trajectory.size(2) == 7: # Check only for minigrid (vizdoom instruction are different)
                    n_correct_attrib = self._cheat_check(true_mission=hindsight_mission,
                                                         generated_mission=generated_hindsight_mission)
                    self.logger.log("gen/n_correct_attrib_gen", n_correct_attrib)


                score_bleu = bleu_score.sentence_bleu(references=[hindsight_mission],
                                                      hypothesis=generated_hindsight_mission.numpy(),
                                                      smoothing_function=bleu_score.SmoothingFunction().method2,
                                                      weights=(0.5,0.5))

                bleu1 = bleu_score.sentence_bleu(references=[hindsight_mission],
                                                 hypothesis=generated_hindsight_mission.numpy(),
                                                 smoothing_function=bleu_score.SmoothingFunction().method2,
                                                 weights=[1])


                self.logger.log("gen/bleu2", score_bleu)
                self.logger.log("gen/bleu1", bleu1)
                self.generator_usage += 1

                # Substitute the old mission with the new one, change the reward at the end of episode
                hindsight_episode = []
                len_episode = len(self.current_episode)
                for step, (st, a, wrong_reward, st_plus1, end_ep, wrong_mission, length, gamma) in enumerate(self.current_episode):
                    if step >= len_episode - self.n_step:
                        hindsight_reward = 1 * self.gamma ** (len(self.current_episode) - step - 1)
                    else:
                        hindsight_reward = 0
                    len_mission = torch.LongTensor([len(generated_hindsight_mission)])
                    hindsight_episode.append(
                        self.transition(st, a, hindsight_reward, st_plus1, end_ep, generated_hindsight_mission, len_mission, gamma)
                    )
                self._store_episode(hindsight_episode)

            # If the agent succeeded, store the state/instruction pair to train the generator
            elif reward > 0:
                trajectory_to_predict = [self.compressor.decompress_elem(ep.current_state)
                                         for ep in self.current_episode[-self.n_state_to_predict_instruction:]]

                len_traj = len(trajectory_to_predict)
                if len_traj < self.n_state_to_predict_instruction:
                    trajectory_to_predict = [self.dummy_state] * (self.n_state_to_predict_instruction - len_traj) + trajectory_to_predict

                self.generator_dataset["states"].append(torch.cat(trajectory_to_predict, dim=0).unsqueeze(0))
                self.generator_dataset["instructions"].append(mission)
                self.generator_dataset["lengths"].append(mission.size(0))
                self.generator_dataset["correct_obj_name"].append(correct_obj_name)
                # if len(self.generator_dataset["states"]) % 10 == 0:
                #     pkl.dump(self.generator_dataset, open("saved_tools/generator_dataset7.pkl", "wb"))

            self._store_episode(self.current_episode)
            self.current_episode = []

    def _train_generator(self):

        convergence = False
        len_dataset = len(self.generator_dataset["states"])
        # To speed batching, convert list to tensor
        lengths = self.generator_dataset['lengths']
        instructions = self.generator_dataset["instructions"]
        states = self.generator_dataset['states']

        states, instructions, lengths = zip(*sorted(zip(states, instructions, lengths),
                                            key=lambda x: -x[0].size(0)))

        states = torch.cat(states, dim=0)
        lengths = torch.LongTensor(lengths)
        instructions = torch.nn.utils.rnn.pad_sequence(sequences=instructions,
                                                       batch_first=True,
                                                       padding_value=self.padding_value
                                                       )

        losses = []
        accuracies = []
        gen_update_this_epoch = 1
        starting_time = time.time()

        self.instruction_generator.train()
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

            # Last round of test, check gradients and size
            assert logits.requires_grad is True
            assert instruction_label.requires_grad is False
            assert instruction_label.size(0) == logits.size(0)

            loss = self.loss(input=logits, target=instruction_label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.logger and self.n_update_generator % 5000 == 0:
                self.logger.add_scalar("gen/generator_loss", loss.detach().item(), self.n_update_generator)
                self.logger.add_scalar("gen/generator_accuracy", accuracy, self.n_update_generator)

            self.n_update_generator += 1
            gen_update_this_epoch += 1
            losses.append(loss.item())
            accuracies.append(accuracy)

            # Convergence test
            convergence_sample = 10
            if len(losses) > convergence_sample:
                convergence = gen_update_this_epoch > self.max_steps_optim \
                              or np.mean(accuracies[-10:]) > self.accuracy_convergence

            if self.n_update_generator % 50 == 1:
                time_passed = time.time() - starting_time
                print("Iter #{:05}, loss {:.4f}, accuracy {:.2f}, in {:.2f} ({:.3f} per step)".format(
                    self.n_update_generator, losses[-1], accuracy, time_passed, time_passed / gen_update_this_epoch))


        print("Done training generator in {} steps\nAccuracy : {} loss : {}".format(self.n_update_generator, accuracy, losses[-1]))
        if self.logger:
            self.logger.add_scalar("gen/generator_loss", loss.detach().item(), self.n_update_generator)
            self.logger.add_scalar("gen/generator_accuracy", accuracy, self.n_update_generator)

class LearntHindsightRecurrentExperienceReplay(RecurrentReplayBuffer, LearntHindsightExperienceReplay):

    def __init__(self, input_shape, n_output, config, device, logger=None):
        self.len_sum = 0
        LearntHindsightExperienceReplay.__init__(self, input_shape=input_shape, n_output=n_output, config=config, device=device, logger=logger)
        self.MIN_SEQ_SIZE = 4

        # Reduce memory footprint by reducing the number of memory cell available
        self.episode_length = np.zeros(self.memory_size // self.MIN_SEQ_SIZE)

        if self.use_prioritization:
            self.prioritize_max_mean_balance = config["prioritize_max_mean_balance"]
            self.prioritize_p = np.zeros(self.memory_size // self.MIN_SEQ_SIZE)
            self.prioritize_p[0] = 1
            del self.id_range  # Not useful in Recurrent Replay
        self.last_position = 0
    def __len__(self):
        return int(self.len_sum)





if __name__ == "__main__":
    pass
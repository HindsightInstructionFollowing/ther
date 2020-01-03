from algo.basedoubledqn import BaseDoubleDQN
import torch
from torch import nn
import torch.nn.functional as F

import random

class RecurrentDQN(BaseDoubleDQN):
    def __init__(self, env, config, logger, visualizer, device='cpu'):
        super().__init__(env, config, logger, visualizer, device)

        self.bppt_size = config["max_bptt"]

        self.burn_in = config["burn_in"] #0 # Hard since sequences can be super short in minigrid and vizdoom
        assert self.burn_in == 0, "Not available at the moment"

        self.fuse_text_before_memory = True

        self.state_padding = torch.zeros(*self.env.observation_space["image"].shape)
        self.action_padding = env.action_space.n + 1
        self.terminal_padding = 1

    def select_action(self, state, ht):

        self.current_epsilon = max(self.epsilon_init - self.total_steps * (self.epsilon_init - self.epsilon_min)
                                   / self.step_exploration, self.epsilon_min)

        if random.random() < self.current_epsilon:
            action = random.choice(range(self.n_actions))

            # Compute ht for next environment step
            q_values, new_ht = self.policy_net(state, ht, seq_len=1)
            q_values = q_values.detach().cpu().numpy()[0]
        else:
            q_values, new_ht = self.policy_net(state, ht, seq_len=1)
            q_values = q_values.detach().cpu().numpy()[0]
            action = int(q_values.argmax())

        self.total_steps += 1
        return action, q_values, new_ht

    def preprocess_state_sequences(self, transitions):
        """
        Subsample, pad and batch states
        """

        batch_state = []
        batch_next_state = []
        batch_mission = []
        batch_terminal = [] # Not useful to pad
        batch_action = [] # Not useful to pad
        state_sequence_lengths = []

        for state_sequences in transitions:

            # Select only samples required for bptt
            if len(state_sequences) >= self.bppt_size:
                # Todo : sample shorter sequences ? To sample epsisode's end more often
                begin_id_rand = random.randint(0, len(state_sequences)-self.bppt_size)
                state_sequences = state_sequences[begin_id_rand:]

            seq_length = len(state_sequences)
            padding_length = self.bppt_size - seq_length

            if padding_length > 0:
                state_sequences.current_state.extend(  [self.state_padding]                *padding_length)
                state_sequences.next_state.extend(     [self.state_padding]                *padding_length)
                state_sequences.mission.extend(        [state_sequences.mission[-1]]       *padding_length)
                state_sequences.mission_length.extend( [state_sequences.mission_length[-1]]*padding_length)

                # state_sequences.actions.extend([self.action_padding]*padding_length)
                # state_sequences.terminal.extend([self.terminal_padding]*padding_length)

            state_sequence_lengths.append(seq_length)
            batch_state.extend(state_sequences.current_state)
            batch_next_state.extend(state_sequences.next_state)
            batch_terminal.extend(state_sequences.terminal)
            batch_action.extend(state_sequences.action)
            batch_mission.extend(state_sequences.mission)

        return batch_state, batch_next_state, batch_terminal, batch_action, batch_mission, state_sequence_lengths

    def optimize_model(self, state, action, next_state, reward, done):
        hindsight_mission = next_state["hindsight_mission"] if "hindsight_mission" in next_state else None
        self.replay_buffer.add_transition(current_state=state["image"].cpu(),
                                          action=action,
                                          next_state=next_state["image"].cpu(),
                                          reward=reward,
                                          mission=next_state["mission"][0].cpu(),
                                          mission_length=next_state["mission_length"].cpu(),
                                          terminal=done,
                                          hindsight_mission=hindsight_mission)

        if len(self.replay_buffer) < self.batch_size:
            return 0

        # Sample from the memory replay
        transitions = self.replay_buffer.sample(self.batch_size)

        batch_state, batch_next_state, batch_terminal, batch_action, batch_mission, state_sequence_lengths = self.pad_state_sequences(transitions=transitions)
        batch_curr_state = torch.cat(batch_state).to(device=self.device)
        batch_next_state = torch.cat(batch_next_state).to(device=self.device)

        batch_terminal = torch.as_tensor(batch_terminal, dtype=torch.int32, device=self.device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.long, device=self.device).reshape(-1, 1)
        #batch_mission_length = torch.cat(batch_mission_length).to(self.device)


        quit()

        ###########################################################################################################
        ###########################################################################################################
        ###########################################################################################################
        ###########################################################################################################
        ###########################################################################################################
        ###########################################################################################################
        ###########################################################################################################


        # Sort transitions by missions length (for packing and padding)
        transitions = sorted(transitions,
                             key=lambda x: -x.mission.size(0))

        # Batch the transitions into one namedtuple
        batch_transitions = self.replay_buffer.transition(*zip(*transitions))

        # Create batches data, easier to manipulate
        batch_curr_state = torch.cat(batch_transitions.current_state).to(device=self.device)
        batch_next_state = torch.cat(batch_transitions.next_state).to(device=self.device)
        batch_terminal = torch.as_tensor(batch_transitions.terminal, dtype=torch.int32, device=self.device)
        batch_action = torch.as_tensor(batch_transitions.action, dtype=torch.long, device=self.device).reshape(-1, 1)
        batch_mission_length = torch.cat(batch_transitions.mission_length).to(self.device)

        batch_mission = nn.utils.rnn.pad_sequence(sequences=batch_transitions.mission,
                                                  batch_first=True,
                                                  padding_value=2 # Padding is always 2, checked by vocab
                                                  ).to(self.device)

        # Compute targets according to the Bellman eq
        batch_next_state_non_terminal_dict = {
            "image": batch_next_state[batch_terminal == 0],
            "mission": batch_mission[batch_terminal == 0],
            "mission_length": batch_mission_length[batch_terminal == 0]
        }

        # Evaluation of the Q value with the target net
        targets = torch.as_tensor(batch_transitions.reward, dtype=torch.float32, device=self.device).reshape(-1, 1)

        # Double DQN
        if torch.sum(batch_terminal) != self.batch_size:
            # Selection of the action with the policy net
            q_values, _, _ = self.policy_net(batch_next_state_non_terminal_dict)
            q_values_next_state, _, _ = self.target_net(batch_next_state_non_terminal_dict)

            args_actions = q_values.max(1)[1].reshape(-1, 1)
            targets[batch_terminal == 0] = targets[batch_terminal == 0] \
                                       + self.gamma \
                                       * q_values_next_state.gather(1, args_actions).detach()

        targets = targets.reshape(-1)

        # Compute the current estimate of Q
        batch_curr_state_dict = {
            "image": batch_curr_state,
            "mission": batch_mission,
            "mission_length": batch_mission_length
        }
        predictions, _, _ = self.policy_net(batch_curr_state_dict)
        predictions = predictions.gather(1, batch_action).view(-1)

        # Loss
        loss = F.smooth_l1_loss(predictions, targets)
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        for name, param in self.policy_net.named_parameters():
            if hasattr(param.grad, 'data'):
                param.grad.data.clamp_(-1, 1)

        # self.old_parameters = dict()
        # for k, v in self.target_net.state_dict().items():
        #     self.old_parameters[k] = v.cpu()

        # Do the gradient descent step
        self.optimizer.step()

        if self.environment_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.n_update_target += 1

        # self.new_parameters = dict()
        # for k,v in self.target_net.state_dict().items():
        #     self.new_parameters[k] = v.cpu()
        # self.check_weigths_change()

        # Log important info, see logging_helper => SweetLogger for more details
        if self.writer:
            self.writer.log("percent_terminal", batch_terminal.sum().item()/self.batch_size)
            self.writer.log("n_update_target", self.n_update_target)

        return loss.detach().item()





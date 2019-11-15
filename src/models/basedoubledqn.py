import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.neural_architecture import MinigridConv, MlpNet
from models.replay_buffer import ReplayMemory

class BaseDoubleDQN(nn.Module):

    #def __init__(self, h, w, c, n_actions, frames, lr, num_token, device, use_memory, use_text):
    def __init__(self, obs_space, action_space, config, writer=None):
        """
        h: height of the screen
        w: width of the screen
        frames: last observations to make a state
        n_actions: number of actions
        lr: learning rate
        num_token: number of words, useful only for the onehot modelisation
        device: device to use
        use_memory: boolean, 1: the frames are processed with a LSTM, 0: the frames are stacked to make a state
        """
        super(BaseDoubleDQN, self).__init__()

        if config["dqn_architecture"] == "conv":
            nn_creator = lambda : MinigridConv(obs_space=obs_space, action_space=action_space, use_lstm_after_conv=True)
        else:
            nn_creator = lambda : MlpNet(obs_space=obs_space, action_space=action_space)

        self.policy_net = nn_creator()
        self.target_net = nn_creator()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=config["lr"])

        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.n_actions = action_space.n

        self.replay_buffer = ReplayMemory(size=config["replay_buffer_size"])

        self.epsilon_init = 1
        self.epsilon_min = 0.04
        self.step_exploration = config["step_exploration"]
        self.current_epsilon = self.epsilon_init
        self.total_steps = 0

        self.update_target_every = config["update_target_every"]
        self.n_update_target = 0

        self.device = config["device"]
        self.to(self.device)

    def select_action(self, state):

        self.current_epsilon = max(self.epsilon_init - self.total_steps * (self.epsilon_init - self.epsilon_min)
                                   / self.step_exploration, self.epsilon_min)

        if random.random() < self.current_epsilon:
            action = random.choice(range(self.n_actions))
            q_values = [ 1 / self.n_actions for i in range(self.n_actions)]
        else:
            # max(1) for the dim, [1] for the indice, [0] for the value
            copy_state = state.copy()
            copy_state["text_length"] = [state["mission"].shape[0]]
            copy_state["mission"] = state["mission"].unsqueeze(0)

            q_values = self.policy_net(copy_state).detach().cpu().numpy()[0]
            action = int(q_values.argmax())

        self.total_steps += 1
        return action, q_values

    def end_of_episode(self, n_episodes):
        pass

    def store_transitions(self, state, action, next_state, reward, done, mission, use_hindsight=False):
        self.replay_buffer.add_transition(curr_state=state,
                                          action=action,
                                          reward=reward,
                                          next_state=next_state,
                                          mission=mission,
                                          terminal=done)

    # Optimize the model
    def optimize_model(self, state, action, next_state, reward, done, mission, environment_step):

        self.store_transitions(state=state,
                               action=action,
                               next_state=next_state,
                               reward=reward,
                               mission=mission,
                               done=done)

        if len(self.replay_buffer) < self.batch_size:
            return 0

        # Sample from the memory replay
        transitions = self.replay_buffer.sample(self.batch_size)
        # Batch the transitions into one namedtuple

        batch_transitions = self.replay_buffer.transition(*zip(*transitions))
        batch_curr_state = torch.cat(batch_transitions.curr_state)
        batch_next_state = torch.cat(batch_transitions.next_state)
        batch_terminal = torch.as_tensor(batch_transitions.terminal, dtype=torch.int32)
        batch_action = torch.as_tensor(batch_transitions.action, dtype=torch.long, device=self.device).reshape(-1, 1)

        text_length = [None] * self.batch_size
        for ind, mission in enumerate(batch_transitions.mission):
            text_length[ind] = mission.size(0)
        batch_text_length = torch.tensor(text_length, dtype=torch.long).to(self.device)
        batch_mission = nn.utils.rnn.pad_sequence(batch_transitions.mission, batch_first=True).to(self.device)

        # Compute targets according to the Bellman eq
        batch_next_state_non_terminal_dict = {
            "image": batch_next_state[batch_terminal == 0],
            "mission": batch_mission[batch_terminal == 0],
            "text_length": batch_text_length[batch_terminal == 0]
        }

        # Evaluation of the Q value with the target net
        targets = torch.as_tensor(batch_transitions.reward, dtype=torch.float32, device=self.device).reshape(-1, 1)

        # Double DQN
        if torch.sum(batch_terminal) != self.batch_size:
            # Selection of the action with the policy net
            args_actions = self.policy_net(batch_next_state_non_terminal_dict).max(1)[1].reshape(-1, 1)
            targets[batch_terminal == 0] = targets[batch_terminal == 0] \
                                       + self.gamma \
                                       * self.target_net(batch_next_state_non_terminal_dict).gather(1, args_actions).detach()

        targets = targets.reshape(-1)

        # Compute the current estimate of Q
        batch_curr_state_dict = {
            "image": batch_curr_state,
            "mission": batch_mission,
            "text_length": batch_text_length
        }
        predictions = self.policy_net(batch_curr_state_dict).gather(1, batch_action).view(-1)

        # Loss
        loss = F.smooth_l1_loss(predictions, targets)
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        for name, param in self.policy_net.named_parameters():
            param.grad.data.clamp_(-1, 1)

        # self.old_parameters = dict()
        # for k, v in self.policy_net.state_dict().items():
        #     self.old_parameters[k] = v.cpu()

        # Do the gradient descent step
        self.optimizer.step()

        # self.new_parameters = dict()
        # for k,v in self.policy_net.state_dict().items():
        #     self.new_parameters[k] = v.cpu()
        #
        # self.check_weigths_change()

        if environment_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.n_update_target += 1

        return loss.detach().item()

    def check_weigths_change(self):
        for param_name in self.old_parameters:
            assert not torch.equal(self.new_parameters[param_name], self.old_parameters[param_name]),\
                "param {} didn't change".format(param_name)



import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.neural_architecture import MinigridConv, MlpNet
from algo.replay_buffer import ReplayMemory

import time

class BaseDoubleDQN(nn.Module):

    #def __init__(self, h, w, c, n_actions, frames, lr, num_token, device, use_memory, use_text):
    def __init__(self, env, config, logger, visualizer, device='cpu'):
        """
        """
        super(BaseDoubleDQN, self).__init__()

        if config["architecture"] == "conv":
            nn_creator = MinigridConv
        else:
            nn_creator = MlpNet

        self.env = env
        self.tf_logger = logger
        self.q_values_visualizer = visualizer

        self.policy_net = nn_creator(obs_space=env.observation_space, action_space=env.action_space, config=config["architecture_params"])
        self.target_net = nn_creator(obs_space=env.observation_space, action_space=env.action_space, config=config["architecture_params"])
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(),
                                             lr=config["lr"],
                                             weight_decay=config["weight_decay"])

        self.batch_size = config["batch_size"]
        self.gamma = config["gamma"]
        self.n_actions = env.action_space.n

        self.replay_buffer = ReplayMemory(size=config["replay_buffer_size"])
        self.use_her = config["use_her"]

        self.epsilon_init = 1
        self.epsilon_min = 0.04
        self.step_exploration = config["step_exploration"]
        self.current_epsilon = self.epsilon_init
        self.total_steps = 0

        self.update_target_every = config["update_target_every"]
        self.n_update_target = 0

        self.device = device
        self.to(self.device)

        self.writer = logger

    def select_action(self, state):

        self.current_epsilon = max(self.epsilon_init - self.total_steps * (self.epsilon_init - self.epsilon_min)
                                   / self.step_exploration, self.epsilon_min)

        if random.random() < self.current_epsilon:
            action = random.choice(range(self.n_actions))
            q_values = [ 1 / self.n_actions for i in range(self.n_actions)]
        else:
            # max(1) for the dim, [1] for the indice, [0] for the value
            q_values, v, _ = self.policy_net(state)
            q_values = q_values.detach().cpu().numpy()[0]
            action = int(q_values.argmax())

        self.total_steps += 1
        return action, q_values

    def end_of_episode(self, n_episodes):
        pass

    def store_transitions(self, state, action, next_state, reward, done, mission, mission_length=None, use_hindsight=False):
        self.replay_buffer.add_transition(curr_state=state.cpu(),
                                          action=action,
                                          reward=reward,
                                          next_state=next_state.cpu(),
                                          mission=mission.cpu(),
                                          mission_length=mission_length.cpu(),
                                          terminal=done)

    # Optimize the model
    def optimize_model(self, state, action, next_state, reward, done):

        self.store_transitions(state=state["image"],
                               action=action,
                               next_state=next_state["image"],
                               reward=reward,
                               mission=next_state["mission"],
                               mission_length=next_state["mission_length"],
                               done=done)

        if len(self.replay_buffer) < self.batch_size:
            return 0

        # Sample from the memory replay
        transitions = self.replay_buffer.sample(self.batch_size)
        # Batch the transitions into one namedtuple

        batch_transitions = self.replay_buffer.transition(*zip(*transitions))
        batch_curr_state = torch.cat(batch_transitions.curr_state).to(device=self.device)
        batch_next_state = torch.cat(batch_transitions.next_state).to(device=self.device)
        batch_terminal = torch.as_tensor(batch_transitions.terminal, dtype=torch.int32, device=self.device)
        batch_action = torch.as_tensor(batch_transitions.action, dtype=torch.long, device=self.device).reshape(-1, 1)
        batch_mission_length = torch.cat(batch_transitions.mission_length)

        # text_length = [None] * self.batch_size
        # for ind, mission in enumerate(batch_transitions.mission):
        #     text_length[ind] = mission.size(0)
        # batch_mission_length = torch.tensor(text_length, dtype=torch.long).to(self.device)
        batch_mission = nn.utils.rnn.pad_sequence(batch_transitions.mission, batch_first=True).to(self.device)

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

        if self.writer:
            self.writer.log("percent_terminal", batch_terminal.sum().item()/self.batch_size)
            self.writer.log("n_update_target", self.n_update_target)

        return loss.detach().item()

    def check_weigths_change(self):
        for param_name in self.old_parameters:
            assert torch.equal(self.new_parameters[param_name], self.old_parameters[param_name]),\
                "param {} changed".format(param_name)


    def train(self, n_env_iter, visualizer=None, display=None):

        # todo Some gym self.env require a fake display
        if not display:
            display = open("empty_context.txt", 'w')
        self.environment_step = 1
        episode_num = 1

        with display:
            while self.environment_step < n_env_iter:

                done = False
                obs = self.env.reset()
                iter_this_ep = 0
                reward_this_ep = 0
                begin_ep_time = time.time()

                while not done:
                    act, q_values = self.select_action(obs)
                    new_obs, reward, done, info = self.env.step(act)

                    iter_this_ep += 1
                    self.environment_step += 1
                    reward_this_ep += reward

                    loss = self.optimize_model(state=obs,
                                               action=act,
                                               next_state=new_obs,
                                               reward=reward,
                                               done=done
                                               )

                    obs = new_obs

                    self.tf_logger.log("loss", loss)
                    self.tf_logger.log("max_q_val", max(q_values), operation='max')
                    self.tf_logger.log("min_q_val", min(q_values), operation='min')

                    # Dump tensorboard stats
                    self.tf_logger.dump(total_step=self.environment_step)

                    # Dump image
                    image = self.q_values_visualizer.render_state_and_q_values(game=self.env, q_values=q_values,
                                                                               ep_num=episode_num)
                    if image is not None:
                        self.tf_logger.add_image(tag="data/q_value_ep{}".format(episode_num),
                                            img_tensor=image,
                                            global_step=iter_this_ep,
                                            dataformats="HWC")

                # ============ END OF EP ==============
                # =====================================
                episode_num += 1
                time_since_ep_start = time.time() - begin_ep_time

                loss_mean = np.mean(self.tf_logger.variable_to_log['loss']['values'])
                print("loss_mean {}".format(loss_mean))
                print(
                "End of ep #{} Time since begin ep : {:.2f}, Time per step : {:.2f} Total iter : {}  iter this ep : {} rewrd : {:.3f}".format(
                    episode_num, time_since_ep_start, time_since_ep_start / iter_this_ep, self.environment_step, iter_this_ep,
                    reward_this_ep))

                self.tf_logger.log("n_iter_per_ep", iter_this_ep)
                self.tf_logger.log("wrong_pick", int(iter_this_ep < self.env.unwrapped.max_steps and reward_this_ep <= 0))
                self.tf_logger.log("time_out", int(iter_this_ep >= self.env.unwrapped.max_steps))
                self.tf_logger.log("reward", reward_this_ep)
                self.tf_logger.log("accuracy", reward_this_ep > 0)
                self.tf_logger.log("epsilon", self.current_epsilon)




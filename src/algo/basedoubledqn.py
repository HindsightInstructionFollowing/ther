import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algo.neural_architecture import MinigridConvPolicy, MlpNet, MinigridRecurrentPolicy
from algo.replay_buffer_parallel import ReplayBufferParallel

from algo.learnt_her import LearntHindsightExperienceReplay

from algo.transition import basic_transition

import time

class BaseDoubleDQN(nn.Module):

    def __init__(self, env, config, test_env=None, logger=None, visualizer=None, device='cpu'):
        super(BaseDoubleDQN, self).__init__()

        if config["architecture"] == "conv":
            nn_creator = MinigridConvPolicy
        elif config["architecture"] == "conv_lstm":
            nn_creator = MinigridRecurrentPolicy
        else:
            nn_creator = MlpNet

        self.env = env
        self.test_env = test_env
        self.tf_logger = logger
        self.q_values_visualizer = visualizer

        self.policy_net = nn_creator(obs_space=env.observation_space, action_space=env.action_space, config=config["architecture_params"], device=device)
        self.target_net = nn_creator(obs_space=env.observation_space, action_space=env.action_space, config=config["architecture_params"], device=device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimize = config["optimize"] if "optimize" in config else True
        if config["optimizer"].lower() == 'rmsprop':
            self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(),
                                                 lr=config["lr"],
                                                 weight_decay=config["weight_decay"])
        elif config["optimizer"].lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.policy_net.parameters(),
                                              lr=config["lr"],
                                              weight_decay=config["weight_decay"])
        else:
            raise NotImplementedError("Optimizer '{}' doesn't exist".format(config["optimizer"].lower()))

        self.wait_steps_before_optim = config["wait_steps_before_optim"]
        self.n_optimize_per_step = config["n_optimize_per_step"]
        self.n_update_policy_net = 0

        self.batch_size = config["experience_replay_config"]["batch_size"]
        self.n_actions = env.action_space.n

        # Setting gamma and configuring n-step estimations
        config["experience_replay_config"]["n_step"] = config["n_step"]
        config["experience_replay_config"]["gamma"] = config["gamma"]

        is_recurrent = config["architecture"] == "conv_lstm"

        # Create replay buffer here
        if config["experience_replay_config"]["use_ther"]:
            replay_buffer = LearntHindsightExperienceReplay(config=config["experience_replay_config"],
                                                            is_recurrent=is_recurrent,
                                                            env=env,
                                                            device=device,
                                                            logger=logger
                                                            )
        else:
            replay_buffer = ReplayBufferParallel(config=config["experience_replay_config"],
                                                 is_recurrent=is_recurrent,
                                                 env=env
                                                 )

        self.replay_buffer = replay_buffer

        self.epsilon_init = 1
        self.epsilon_min = 0.04
        self.step_exploration = config["step_exploration"]
        self.current_epsilon = self.epsilon_init
        self.total_steps = 0

        self.grad_norm_limit = config["grad_norm_limit"]
        self.update_target_every = config["update_target_every"]
        self.n_update_target = 0

        self.device = device
        self.to(self.device)

        self.writer = logger
        self.PADDING_MISSION = 2  # Padding is always 2, checked by vocab

    def select_action(self, state, ht=None):
        self.current_epsilon = max(self.epsilon_init - self.total_steps * (self.epsilon_init - self.epsilon_min)
                                   / self.step_exploration, self.epsilon_min)

        if random.random() < self.current_epsilon:
            action = random.choice(range(self.n_actions))
            q_values = [ 1 / self.n_actions for i in range(self.n_actions)]
        else:
            q_values, v, _ = self.policy_net(state)
            q_values = q_values.detach().cpu().numpy()[0]
            action = int(q_values.argmax())

        self.total_steps += 1
        new_ht = None
        return action, q_values, new_ht

    def end_of_episode(self, n_episodes):
        pass

    # Optimize the model
    def optimize_model(self, state, action, next_state, reward, done):

        hindsight_mission = next_state["hindsight_mission"] if "hindsight_mission" in next_state else None
        object_name = next_state["correct_obj_name"] if "correct_obj_name" in next_state else None

        self.replay_buffer.add_transition(current_state=state["image"].cpu(),
                                          action=action,
                                          next_state=next_state["image"].cpu(),
                                          reward=reward,
                                          mission=next_state["mission"][0].cpu(),
                                          mission_length=next_state["mission_length"].cpu(),
                                          terminal=done,
                                          hindsight_mission=hindsight_mission,
                                          correct_obj_name=object_name)

        if len(self.replay_buffer) < self.batch_size:
            return 0

        # Sample from the memory replay
        transitions, is_weights = self.replay_buffer.sample(self.batch_size)

        # Sort transitions by missions length (for packing and padding)
        zipped = zip(transitions, is_weights)
        transitions, is_weights = zip(*sorted(zipped,
                                             key=lambda x: -x[0].mission.size(0))
                                      )
        # Batch the transitions into one namedtuple
        batch_transitions = basic_transition(*zip(*transitions))

        # Create batches data, easier to manipulate
        batch_curr_state = torch.cat(batch_transitions.current_state).to(device=self.device)
        batch_next_state = torch.cat(batch_transitions.next_state).to(device=self.device)
        batch_terminal = torch.as_tensor(batch_transitions.terminal, dtype=torch.int32, device=self.device)
        batch_action = torch.as_tensor(batch_transitions.action, dtype=torch.long, device=self.device).reshape(-1, 1)
        batch_mission_length = torch.cat(batch_transitions.mission_length).to(self.device)
        batch_gamma = torch.FloatTensor(batch_transitions.gamma).to(self.device) # For n-step gamma might vary a bit
        is_weights = torch.FloatTensor(is_weights).to(device=self.device)

        batch_mission = nn.utils.rnn.pad_sequence(sequences=batch_transitions.mission,
                                                  batch_first=True,
                                                  padding_value=self.PADDING_MISSION
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
            terminal_index = batch_terminal == 0
            #assert targets[terminal_index].sum() == 0

            # Selection of the action with the policy net
            q_values, _, _ = self.policy_net(batch_next_state_non_terminal_dict)
            q_values_next_state, _, _ = self.target_net(batch_next_state_non_terminal_dict)

            args_actions = q_values.max(1)[1].reshape(-1, 1)
            targets[terminal_index,0] = targets[terminal_index].view(-1) \
                                       + batch_gamma[terminal_index] * q_values_next_state.gather(1, args_actions).view(-1).detach()

        targets = targets.reshape(-1)

        # Compute the current estimate of Q
        batch_curr_state_dict = {
            "image": batch_curr_state,
            "mission": batch_mission,
            "mission_length": batch_mission_length
        }
        predictions, _, _ = self.policy_net(batch_curr_state_dict)
        predictions = predictions.gather(1, batch_action).view(-1)

        # Update prio
        delta = torch.abs(predictions - targets).detach().cpu().numpy()
        self.replay_buffer.update_transitions_proba(delta)

        # Loss
        loss = F.smooth_l1_loss(predictions, targets, reduction='none') * is_weights
        loss = loss.mean()
        # Optimization
        self.optimizer.zero_grad()
        loss.backward()

        # Keep the gradient between (-1,1). Works like one uses L1 loss for large gradients (see Huber loss)
        nn.utils.clip_grad_norm(self.policy_net.parameters(), self.grad_norm_limit)
        # for name, param in self.policy_net.named_parameters():
        #     if hasattr(param.grad, 'data'):
        #         param.grad.data.clamp_(-self.grad_norm_limit, self.grad_norm_limit)

        # self.old_parameters = dict()
        # for k, v in self.target_net.state_dict().items():
        #     self.old_parameters[k] = v.cpu()

        # Do the gradient descent step
        self.optimizer.step()
        self.n_update_policy_net += 1

        if self.n_update_policy_net % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.n_update_target += 1

        # self.new_parameters = dict()
        # for k,v in self.target_net.state_dict().items():
        #     self.new_parameters[k] = v.cpu()
        # self.check_weigths_change()

        # Log important info, see logging_helper => SweetLogger for more details
        if self.writer:
            self.writer.store_buffer_id(self.replay_buffer.last_id_sampled)
            self.writer.log("train/percent_terminal", batch_terminal.sum().item()/self.batch_size)
            self.writer.log("train/n_update_target", self.n_update_target)

        return loss.detach().item()

    def check_weigths_change(self):
        for param_name in self.old_parameters:
            assert torch.equal(self.new_parameters[param_name], self.old_parameters[param_name]),\
                "param {} changed".format(param_name)


    def test(self):

        print("==============================")
        print("Break, using model in test env")
        print("==============================")

        test_step = 0
        episode_num = 0

        while test_step < self.test_env.n_step_test:

            done = False
            obs = self.test_env.reset()
            iter_this_ep = 0
            reward_this_ep = 0
            begin_ep_time = time.time()
            ht = None

            while not done:
                act, q_values, ht = self.select_action(obs, ht)
                new_obs, reward, done, info = self.test_env.step(act)

                iter_this_ep += 1
                test_step += 1
                reward_this_ep += reward

                obs = new_obs

                self.tf_logger.log("test/max_q_val", max(q_values), operation='max')
                self.tf_logger.log("test/min_q_val", min(q_values), operation='min')

                # Dump tensorboard stats
                self.tf_logger.dump(total_step=test_step)

            # ============ END OF EP ==============
            # =====================================
            episode_num += 1
            time_since_ep_start = time.time() - begin_ep_time

            print(
                "TEST : End of ep #{} Time since begin ep : {:.2f}, Time per step : {:.2f} Total iter : {}  iter this ep : {} rewrd : {:.3f}".format(
                    episode_num, time_since_ep_start, time_since_ep_start / iter_this_ep, test_step,
                    iter_this_ep,
                    reward_this_ep))

            self.tf_logger.log("test/n_iter_per_ep", iter_this_ep)
            self.tf_logger.log("test/wrong_pick", int(iter_this_ep < self.env.unwrapped.max_steps and reward_this_ep <= 0))
            self.tf_logger.log("test/time_out", int(iter_this_ep >= self.env.unwrapped.max_steps))
            self.tf_logger.log("test/reward", reward_this_ep)
            self.tf_logger.log("test/accuracy", reward_this_ep > 0)

        print("Back to training")
        print("================")

    def train(self, n_env_iter, visualizer=None):

        self.environment_step = 1
        episode_num = 1

        next_test = self.test_env.n_step_between_test if self.test_env is not None else 0
        while self.environment_step < n_env_iter:

            done = False
            obs = self.env.reset()
            iter_this_ep = 0
            reward_this_ep = 0
            begin_ep_time = time.time()
            ht = None

            while not done:
                act, q_values, ht = self.select_action(obs, ht)
                new_obs, reward, done, info = self.env.step(act)
                iter_this_ep += 1
                self.environment_step += 1
                reward_this_ep += reward

                for up in range(self.n_optimize_per_step):
                    loss = self.optimize_model(state=obs,
                                               action=act,
                                               next_state=new_obs,
                                               reward=reward,
                                               done=done
                                               )
                    self.tf_logger.log("train/loss", loss)


                obs = new_obs

                self.tf_logger.log("train/max_q_val", max(q_values), operation='max')
                self.tf_logger.log("train/min_q_val", min(q_values), operation='min')

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

            loss_mean = np.mean(self.tf_logger.variable_to_log['train/loss']['values'])
            print("loss_mean {}".format(loss_mean))
            print(
            "End of ep #{} Time since begin ep : {:.2f}, Time per step : {:.2f} Total iter : {}  iter this ep : {} rewrd : {:.3f}".format(
                episode_num, time_since_ep_start, time_since_ep_start / iter_this_ep, self.environment_step, iter_this_ep,
                reward_this_ep))

            self.tf_logger.log("train/n_iter_per_ep", iter_this_ep)
            self.tf_logger.log("train/wrong_pick", int(iter_this_ep < self.env.unwrapped.max_steps and reward_this_ep <= 0))
            self.tf_logger.log("train/time_out", int(iter_this_ep >= self.env.unwrapped.max_steps))
            self.tf_logger.log("train/reward", reward_this_ep)
            self.tf_logger.log("train/accuracy", reward_this_ep > 0)
            self.tf_logger.log("train/epsilon", self.current_epsilon)

            if self.test_env and self.environment_step > next_test:
                self.test()
                next_test = self.environment_step + self.test_env.n_step_between_test




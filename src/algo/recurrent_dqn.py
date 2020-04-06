from algo.basedoubledqn import BaseDoubleDQN
import torch
from torch import nn
import torch.nn.functional as F

import random
from os import path
import numpy as np

class RecurrentDQN(BaseDoubleDQN):
    def __init__(self, env, config, logger=None, visualizer=None, test_env=None, device='cpu'):
        super().__init__(env=env, test_env=test_env, config=config, logger=logger, visualizer=visualizer, device=device)
        self.fuse_text_before_memory = True

    def select_action(self, state, ht):

        self.current_epsilon = max(self.epsilon_init - self.total_steps * (self.epsilon_init - self.epsilon_min)
                                   / self.step_exploration, self.epsilon_min)

        state["max_sequence_length"] = 1
        state["state_sequence_lengths"] = torch.ones(1).to(self.device)
        state["padding_mask"] = torch.ones(1).to(self.device)

        if random.random() < self.current_epsilon:
            action = random.choice(range(self.n_actions))

            # Compute ht for next environment step
            q_values, new_ht = self.policy_net(state, ht)
            q_values = q_values.detach().cpu().numpy()[0]
        else:
            q_values, new_ht = self.policy_net(state, ht)
            q_values = q_values.detach().cpu().numpy()[0]
            action = int(q_values.argmax())

        self.total_steps += 1
        return action, q_values, new_ht.detach()

    def optimize_model(self, state, action, next_state, reward, done):
        hindsight_mission = next_state["hindsight_mission"] if "hindsight_mission" in next_state else None
        object_name = next_state["correct_obj_name"] if "correct_obj_name" in next_state else None

        self.replay_buffer.add_transition(current_state=state["image"].cpu().numpy(),
                                          action=action,
                                          next_state=next_state["image"].cpu().numpy(),
                                          reward=reward,
                                          mission=next_state["mission"][0].cpu(),
                                          mission_length=next_state["mission_length"].cpu(),
                                          terminal=done,
                                          hindsight_mission=hindsight_mission,
                                          correct_obj_name=object_name)

        if self.total_steps < self.wait_steps_before_optim:
            return 0

        # Sample from the memory replay
        batch_dict, is_weights = self.replay_buffer.sample()

        # Padding mask corresponding to real transitions and fake padded one to allows for the lstm computation
        batch_mask =            batch_dict["padding_mask"].view(-1)
        batch_sequence_length = batch_dict["state_sequence_lengths"].view(-1)

        batch_curr_state =      batch_dict["state"].to(device=self.device).view(-1, *self.env.observation_space["image"].shape)
        batch_next_state =      batch_dict["next_state"].to(device=self.device).view(-1, *self.env.observation_space["image"].shape)

        batch_mission =         batch_dict["mission"].to(self.device).view(batch_curr_state.size(0), -1)
        batch_mission_length =  batch_dict["mission_length"].to(self.device).view(-1)

        # Convert to torch and remove padding since it's not useful for those variables
        batch_terminal =        batch_dict["terminal"].view(-1)[batch_mask == 1].to(self.device)
        batch_action_full =     batch_dict["action"].to(device=self.device).view(-1)
        batch_action =          batch_action_full[batch_mask == 1].view(-1, 1)
        batch_last_action =     batch_dict["last_action"].to(device=self.device)
        batch_gamma =           batch_dict["gamma"].type(torch.FloatTensor).view(-1)[batch_mask == 1].view(-1, 1).to(device=self.device)

        is_weights =            torch.FloatTensor([is_weights]).to(self.device)

        #============= Computing targets ===========
        #===========================================
        targets = batch_dict["reward"].type(torch.FloatTensor).view(-1)[batch_mask == 1].to(self.device).reshape(-1, 1)

        batch_next_state_non_terminal_dict = {
            "image": batch_next_state,
            "mission": batch_mission,
            "mission_length": batch_mission_length,
            "last_action" : batch_action_full, # Can be useful for the lstm
            "padding_mask": batch_mask,
            "state_sequence_lengths": batch_sequence_length,
            "max_sequence_length": self.env.unwrapped.max_steps
        }

        # Double DQN : Selection of the action with the policy net
        q_values_for_action, _ = self.policy_net(batch_next_state_non_terminal_dict)
        q_values_next_state, _ = self.target_net(batch_next_state_non_terminal_dict)

        terminal_index = batch_terminal == 0

        q_values_for_action = q_values_for_action[terminal_index]
        q_values_next_state = q_values_next_state[terminal_index]

        args_actions = q_values_for_action.max(1)[1].reshape(-1, 1)
        targets[terminal_index, 0] = targets[terminal_index].view(-1) \
                                      + batch_gamma[terminal_index].view(-1) * q_values_next_state.gather(1, args_actions).view(-1).detach()

        targets = targets.reshape(-1)

        # ====== Compute the current estimate of Q ======
        # ===============================================
        batch_curr_state_dict = {
            "image": batch_curr_state,
            "mission": batch_mission,
            "mission_length": batch_mission_length,
            "last_action" : batch_last_action,
            "padding_mask" : batch_mask,
            "state_sequence_lengths" : batch_sequence_length,
            "max_sequence_length" : self.env.unwrapped.max_steps
        }
        predictions, _ = self.policy_net(batch_curr_state_dict)
        predictions = predictions.gather(1, batch_action).view(-1)

        assert predictions.size(0) == batch_sequence_length.sum()

        # Update prio
        delta = torch.abs(predictions - targets).detach().cpu().numpy()
        self.replay_buffer.update_transitions_proba(errors=delta,
                                                    id_retrieved=batch_dict["id_retrieved"][0],
                                                    verification_id=batch_dict["id_verification"][0])

        # Loss
        loss = F.smooth_l1_loss(predictions, targets, reduction='none') * is_weights
        loss = loss.mean()

        if self.optimize:
            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_norm_limit)
            # Do the gradient descent step
            self.optimizer.step()

        if self.environment_step % self.update_target_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            path_weight = path.join(self.writer.path_to_log, "weights.th")
            torch.save(self.policy_net.state_dict(), path_weight)

            if hasattr(self.replay_buffer.batch_sampler, 'prioritize_p'):
                prioritize_path = path.join(self.writer.path_to_log, "prioritize_p.npy")
                np.save(prioritize_path, self.replay_buffer.batch_sampler.prioritize_p)

            self.n_update_target += 1

        # self.new_parameters = dict()
        # for k,v in self.target_net.state_dict().items():
        #     self.new_parameters[k] = v.cpu()
        # self.check_weigths_change()

        # Log important info, see logging_helper => SweetLogger for more details
        if self.writer:
            self.writer.store_buffer_id(batch_dict["id_retrieved"][0])
            self.writer.log("train/n_update_target", self.n_update_target)
            if self.optimize:
                self.writer.log("train/grad_norm", grad_norm, ['max', 'mean'])

        return loss.detach().item()


if __name__ == "__main__" :

    from gym_minigrid.wrappers import wrap_env_from_list
    from gym_minigrid.envs.fetch_attr import FetchAttrEnv
    import dill

    dqn_config = {
  "name" : "minigrid_fetch_rdddqn_tests_no_text4",
  "algo" : "rdqn",
  "device" : "cuda",

  "dump_log_every" : 10000,

  "algo_params": {
    "architecture" : "conv_lstm",
    "architecture_params" : {
      "ignore_text" : False,
      "rnn_state_hidden_size" : 256,
      "rnn_text_hidden_size" : 128,
      "fc_text_embedding_hidden" : 32,
      "last_hidden_fc_size" : 64
    },

    "experience_replay_config" : {
      "hindsight_reward" : 1,
      "size" : 40000,
      "use_her" : False,
      "prioritize" : False,
      "use_ther" : False,
      "ther_params" : {
        "accuracy_convergence" : 0.9,
        "lr" : 3e-4,
        "batch_size" : 64,
        "weight_decay" : 1e-4,
        "update_steps": [100,300,1000],
        "n_sample_before_using_generator" : 300,

        "architecture_params": {
          "conv_layers_channel" : [16,32,64],
          "conv_layers_size" : [2,2,2],
          "conv_layers_stride" : [1,1,1],
          "max_pool_layers" : [2,0,0],
          "embedding_dim" : 32,
          "generator_max_len" : 10

        }
      }
    },

    "n_parallel_env" : 1,
    "batch_size" : 64,
    "lr" : 1e-5,
    "gamma" : 0.99,
    "update_target_every" : 2000,
    "step_exploration" : 20000,
    "weight_decay" : 0,
    "max_bptt" : 8,
    "burn_in" : 0
  },

  "wrappers_model" : [
    {"name" : "RemoveUselessChannelWrapper", "params" : {}},
    {"name" : "RemoveUselessActionWrapper", "params" : {}},
    {"name" : "MinigridTorchWrapper", "params" : {"device" : "cuda"}}
  ]

}

    env_config = {
            "name": "fetch_4ttr_N10_S10_80_3e6_and_test_env_more_test",
            "gym_name": None,
            "env_type": "fetch",

            "n_env_iter": 3e6,

            "q_visualizer_proba_log": 0,
            "q_visualizer_ep_num_to_log": [1, 2, 100, 1000, 4000],

            "env_params": {
                "missions_file_str": "gym-minigrid/gym_minigrid/envs/missions/fetch_train_missions_80_percent.json",
                "size": 10,
                "numObjs": 10,
                "single_mission": False
            },

            "env_test": {
                "n_step_between_test": 50000,
                "n_step_test": 10000,
                "missions_file_str": "gym-minigrid/gym_minigrid/envs/missions/fetch_holdout_20_percent.json"
            },

            "wrappers_env": [
                {
                    "name": "Word2IndexWrapper",
                    "params": {"vocab_file_str":"gym-minigrid/gym_minigrid/envs/missions/vocab_fetch.json"}
                }
            ]
        }

    env_params = env_config["env_params"]
    env = FetchAttrEnv(size=env_params["size"],
                 numObjs=env_params["numObjs"],
                 missions_file_str=env_params["missions_file_str"],
                 single_mission=env_params["single_mission"])

    wrappers_list_dict = env_config["wrappers_env"]
    wrappers_list_dict.extend(dqn_config["wrappers_model"])

    new_env = wrap_env_from_list(env, wrappers_list_dict)


    dqn = RecurrentDQN(new_env, dqn_config["algo_params"])
    dqn.replay_buffer = dill.load(open("recurrent_buffer_inco.pkl", 'rb'))
    dqn.replay_buffer.len = 3

    dqn.optimize_model()






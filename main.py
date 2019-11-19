# %%
import gym
import numpy as np
from models.basedoubledqn import BaseDoubleDQN
from gym_minigrid.envs.fetch_attr import FetchAttrEnv
from gym_minigrid.wrappers import TextWrapper, FrameStackerWrapper, MinigridTorchWrapper,\
    TorchWrapper, CartPoleWrapper, RemoveUselessActionWrapper, RemoveUselessChannelWrapper
from image_helper import QValueVisualizer

from logging_helper import SweetLogger
import time
import os
import shutil

from xvfbwrapper import Xvfb
#display = Xvfb(width=100, height=100, colordepth=16)
display = open("empty_context.txt", 'w')

# %%


n_frame = 3
device = 'cuda'

config = dict()
config["batch_size"] = 256
config["lr"] = 5e-4
config["replay_buffer_size"] = 40000
config["gamma"] = 0.99
config["device"] = device
config["update_target_every"] = 2000
config["step_exploration"] = 20000
config["weight_decay"] = 0

config["dqn_architecture"] = "conv"
missons_file_str = "gym-minigrid/gym_minigrid/envs/missions/fetch_train_easy_test.json"
env = FetchAttrEnv(size=6,
                   numObjs=3,
                   missions_file_str=missons_file_str)
env = MinigridTorchWrapper(RemoveUselessActionWrapper(TextWrapper(gym.make("MiniGrid-Empty-5x5-v0"))), device=device)
#env = MinigridTorchWrapper(FrameStackerWrapper(LessActionAndObsWrapper(TextWrapper(env=env)), n_stack=n_frame), device=device)

# %%

# config["dqn_architecture"] = "mlp"
# env = TorchWrapper(CartPoleWrapper(gym.make("CartPole-v1")), device=device)

# %%


model = BaseDoubleDQN(obs_space=env.observation_space,
                      action_space=env.action_space,
                      config=config
                      )

n_episodes = 10000
total_step = 1

# SweetLogger is a tensorboardXWriter with additionnal tool to help dealing with lists
expe_path = "out_test/small_minigrid3"
if os.path.exists(expe_path):
    shutil.rmtree(expe_path)
tf_logger = SweetLogger(dump_step=1000, path_to_log=expe_path)

# When do you want to store images of q-function and corresponding state ?
# Specify here :
q_values_generator = QValueVisualizer(proba_log=0.1)
print(env.observation_space)

with display:
    for episode_num in range(n_episodes):

        done = False
        obs = env.reset()
        iter_this_ep = 0
        reward_this_ep = 0
        begin_ep_time = time.time()

        while not done:
            act, q_values = model.select_action(obs)
            new_obs, reward, done, info = env.step(act)

            iter_this_ep += 1
            total_step += 1
            reward_this_ep += reward

            loss = model.optimize_model(state=obs,
                                        action=act,
                                        next_state=new_obs,
                                        reward=reward,
                                        done=done,
                                        environment_step=total_step)

            obs = new_obs

            tf_logger.log("loss", loss)
            tf_logger.log("max_q_val", max(q_values), operation='max')
            tf_logger.log("min_q_val", min(q_values), operation='min')

            # Dump tensorboard stats
            tf_logger.dump(total_step=total_step)

            # Dump image
            image = q_values_generator.render_state_and_q_values(game=env, q_values=q_values, ep_num=episode_num)
            if image is not None:
                tf_logger.add_image(tag="data/q_value_ep{}".format(episode_num),
                                    img_tensor=image,
                                    global_step=iter_this_ep,
                                    dataformats="HWC")

        time_since_ep_start = time.time() - begin_ep_time

        loss_mean = np.mean(tf_logger.variable_to_log['loss']['values'])
        print("loss_mean {}".format(loss_mean))

        # if loss_mean < 0.04:
        #     model.target_net.load_state_dict(model.policy_net.state_dict(), strict=True)
        #     print("Update")


        print("End of ep #{} Time since begin ep : {:.2f}, Time per step : {:.2f} Total iter : {}  iter this ep : {} rewrd : {:.3f}".format(
            episode_num, time_since_ep_start, time_since_ep_start / iter_this_ep, total_step, iter_this_ep, reward_this_ep))

        tf_logger.log("reward", reward_this_ep)
        tf_logger.log("epsilon", model.current_epsilon)



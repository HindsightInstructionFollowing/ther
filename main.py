# %%
import gym
import numpy as np
from models.basedoubledqn import BaseDoubleDQN
from gym_minigrid.envs.fetch_attr import FetchAttrEnv
from gym_minigrid.wrappers import Word2IndexWrapper, FrameStackerWrapper, MinigridTorchWrapper,\
    TorchWrapper, CartPoleWrapper, RemoveUselessActionWrapper, RemoveUselessChannelWrapper, wrap_env_from_list
from image_helper import QValueVisualizer

from logging_helper import SweetLogger
import time
import os
import shutil
from xvfbwrapper import Xvfb

from config import load_config

def train(model_config, env_config, out_dir, seed, model_ext, local_test=None):

    full_config, expe_path = load_config(model_config_file=model_config,
                                         model_ext_file=model_ext,
                                         env_config_file=env_config,
                                         out_dir=out_dir,
                                         seed=seed)

    #todo Some gym env require a fake display
    #display = Xvfb(width=100, height=100, colordepth=16)
    display = open("empty_context.txt", 'w')

    # =================== LOGGING =======================
    # ===================================================

    # SweetLogger is a tensorboardXWriter with additionnal tool to help dealing with lists
    tf_logger = SweetLogger(dump_step=full_config["dump_log_every"], path_to_log=expe_path)

    # When do you want to store images of q-function and corresponding state ?
    # Specify here :
    q_values_generator = QValueVisualizer(proba_log=full_config["q_visualizer_proba_log"],
                                          ep_num_to_log=full_config["q_visualizer_ep_num_to_log"])

    # =================== LOADING ENV =====================
    # =====================================================
    if full_config["gym_name"]:
        env = gym.make(full_config["gym_name"])
    else:
        env_params = full_config["env_params"]

        env = FetchAttrEnv(size=env_params["size"],
                           numObjs=env_params["numObjs"],
                           missions_file_str=env_params["missions_file_str"],
                           single_mission=env_params["single_mission"])

    # =================== APPLYING WRAPPER =====================
    # ==========================================================

    # First, wrappers defined in    env_config
    # Then, wrappers defined in     model_config
    wrappers_list_dict = full_config["wrappers_env"]
    wrappers_list_dict.extend(full_config["wrappers_model"])

    if len(wrappers_list_dict) > 0:
        env = wrap_env_from_list(env, wrappers_list_dict)

    n_episodes = full_config["n_episodes"]
    total_step = 1

    # =================== DEFINE MODEL ========================
    # =========================================================
    model = BaseDoubleDQN(obs_space=env.observation_space,
                          action_space=env.action_space,
                          config=full_config["algo_params"],
                          device=full_config["device"],
                          writer=tf_logger
                          )

    print(env.observation_space)
    # ================ TRAINING HERE ===============
    # ==============================================
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

            # ============ END OF EP ==============
            # =====================================
            time_since_ep_start = time.time() - begin_ep_time

            loss_mean = np.mean(tf_logger.variable_to_log['loss']['values'])
            print("loss_mean {}".format(loss_mean))
            print("End of ep #{} Time since begin ep : {:.2f}, Time per step : {:.2f} Total iter : {}  iter this ep : {} rewrd : {:.3f}".format(
                episode_num, time_since_ep_start, time_since_ep_start / iter_this_ep, total_step, iter_this_ep, reward_this_ep))

            tf_logger.log("n_iter_per_ep", iter_this_ep)
            tf_logger.log("wrong_pick", int(iter_this_ep < env.unwrapped.max_steps and reward_this_ep == 0))
            tf_logger.log("time_out", int(iter_this_ep >= env.unwrapped.max_steps))
            tf_logger.log("reward", reward_this_ep)
            tf_logger.log("epsilon", model.current_epsilon)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-env_config", type=str)
    #parser.add_argument("-env_ext", type=str)
    parser.add_argument("-model_config", type=str)
    parser.add_argument("-model_ext", type=str)
    parser.add_argument("-exp_dir", type=str, default="out", help="Directory all results")
    parser.add_argument("-seed", type=int, default=42, help="Random seed used")
    #parser.add_argument("-local_test", type=bool, default=False, help="If env is run on my PC or a headless server")

    args = parser.parse_args()

    train(env_config=args.env_config,
          #args.env_ext,
          model_config=args.model_config,
          model_ext=args.model_ext,
          out_dir=args.exp_dir,
          seed=args.seed
          )
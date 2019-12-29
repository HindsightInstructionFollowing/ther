# %%
import gym
import numpy as np
from algo.basedoubledqn import BaseDoubleDQN
from algo.ppo import PPOAlgo
from gym_minigrid.envs.fetch_attr import FetchAttrEnv, FetchAttrDictLoaded
from gym_minigrid.wrappers import wrap_env_from_list

from gym_minigrid.envs.relationnal import RelationnalFetch

from env_utils import AttrDict

from image_helper import QValueVisualizer

from logging_helper import SweetLogger
import time
import os
import shutil

from config import load_config
from env_utils import create_doom_env, AttrDict


def train(model_config, env_config, out_dir, seed, model_ext, local_test):

    # =================== CONFIGURATION==================
    # ===================================================
    full_config, expe_path = load_config(model_config_file=model_config,
                                         model_ext_file=model_ext,
                                         env_config_file=env_config,
                                         out_dir=out_dir,
                                         seed=seed)

    # Setting up context, when using a headless server, xvfbwrapper might be necessary
    if not local_test:
        import xvfbwrapper
        display = xvfbwrapper.Xvfb(width=128, height=128, colordepth=16)
    else:
        display = open("empty_context.txt", "r")

        # Override GPU context, switch to CPU on local machine
        full_config["device"] = 'cpu'
        wrapper_gpu = {"name": "MinigridTorchWrapper", "params": {"device": "cuda"}}
        if "wrappers_model" in full_config:
            if wrapper_gpu in full_config["wrappers_model"]:
                index_wrapper = full_config["wrappers_model"].index(wrapper_gpu)
                full_config["wrappers_model"][index_wrapper] = {"name": "MinigridTorchWrapper", "params": {"device": "cpu"}}


    # =================== LOGGING =======================
    # ===================================================

    # SweetLogger is a tensorboardXWriter with additionnal tool to help dealing with lists
    tf_logger = SweetLogger(dump_step=full_config["dump_log_every"], path_to_log=expe_path)

    # When do you want to store images of q-function and corresponding state ?
    # Specify here :
    q_values_visualizer = QValueVisualizer(proba_log=full_config["q_visualizer_proba_log"],
                                          ep_num_to_log=full_config["q_visualizer_ep_num_to_log"])

    # =================== LOADING ENV =====================
    # =====================================================
    if full_config["gym_name"]:
        env_creator = lambda : gym.make(full_config["gym_name"])
    elif full_config["env_type"] == "relationnal":
        env_params = full_config["env_params"]
        env_creator = lambda : RelationnalFetch(size=env_params["size"],
                                                numObjs=env_params["numObjs"],
                                                missions_file_str=env_params["mission_dict_path"])
    elif full_config["env_type"] == "fetch":
        env_params = full_config["env_params"]
        env_creator = lambda : FetchAttrEnv(size=env_params["size"],
                                            numObjs=env_params["numObjs"],
                                            missions_file_str=env_params["missions_file_str"],
                                            single_mission=env_params["single_mission"])
    elif full_config["env_type"] == "vizdoom":
        args = AttrDict(full_config["env_params"])
        env_creator = lambda : create_doom_env(args)
    else:
        env_params = full_config["env_params"]
        env_creator = lambda: FetchAttrDictLoaded(size=env_params["size"],
                                                  numObjs=env_params["numObjs"],
                                                  dict_mission_str=full_config["mission_dict_path"]
                                                  )

    # =================== APPLYING WRAPPER =====================
    # ==========================================================

    # First, wrappers defined in    env_config
    # Then, wrappers defined in     model_config
    wrappers_list_dict = full_config["wrappers_env"]
    wrappers_list_dict.extend(full_config["wrappers_model"])

    envs = []
    for i in range(full_config["algo_params"]["n_parallel_env"]):
        new_env = wrap_env_from_list(env_creator(), wrappers_list_dict)
        envs.append(new_env)

    n_env_iter = full_config["n_env_iter"]

    # =================== DEFINE MODEL ========================
    # =========================================================

    if full_config["algo"] == "dqn":
        model = BaseDoubleDQN(env=envs[0],
                              config=full_config["algo_params"],
                              device=full_config["device"],
                              logger=tf_logger,
                              visualizer=q_values_visualizer,
                              )
    else:
        model = PPOAlgo(envs=envs,
                        config=full_config["algo_params"],
                        logger=tf_logger,
                        visualizer=q_values_visualizer,
                        device=full_config["device"]
                        )

    print(envs[0].observation_space)

    # ================ TRAINING HERE ===============
    model.train(n_env_iter=n_env_iter, visualizer=q_values_visualizer, display=display)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-env_config", type=str)
    #parser.add_argument("-env_ext", type=str)
    parser.add_argument("-model_config", type=str)
    parser.add_argument("-model_ext", type=str)
    parser.add_argument("-exp_dir", type=str, default="out", help="Directory all results")
    parser.add_argument("-seed", type=int, default=42, help="Random seed used")
    parser.add_argument("-local_test", type=bool, default=False, help="If env is run on my PC or a headless server")

    args = parser.parse_args()

    train(env_config=args.env_config,
          #args.env_ext,
          model_config=args.model_config,
          model_ext=args.model_ext,
          out_dir=args.exp_dir,
          seed=args.seed,
          local_test=args.local_test
          )
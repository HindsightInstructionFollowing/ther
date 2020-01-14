#%%

import vizdoom
import argparse
from vizdoom_utils import grounding_env
import numpy as np
import json
from gym_minigrid.wrappers import Vizdoom2Minigrid

#%%

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

#%%

params = json.load(open("config/env/vizdoom_easy.json", 'r'))
args = AttrDict(params["env_params"])

env = grounding_env.GroundingEnv(args)
env.game_init()

env = Vizdoom2Minigrid(env)


num_episodes = 0
rewards_per_episode = []
reward_sum = 0
is_final = 1
while num_episodes < 100:
    if is_final:
        obs = env.reset()
        instruction = obs["mission"]
        print("Instruction: {}".format(instruction))
        print("image.shape", obs["image"].shape)

    # Take a random action
    obs, reward, is_final, _ = \
        env.step(np.random.randint(3))
    reward_sum += reward

    if is_final:
        print("Total Reward: {}".format(reward_sum))
        rewards_per_episode.append(reward_sum)
        num_episodes += 1
        reward_sum = 0
        if num_episodes % 10 == 0:
            print("Avg Reward per Episode: {}".format(
                np.mean(rewards_per_episode)))
#%%


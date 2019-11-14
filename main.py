# %%
import numpy as np
from models.basedoubledqn import BaseDoubleDQN
from gym_minigrid.envs.fetch_attr import FetchAttrEnv
from gym_minigrid.wrappers import LessActionAndObsWrapper, TextWrapper, FrameStackerWrapper, TorchWrapper
from image_helper import QValueVisualizer

from logging_helper import SweetLogger
import time
# %%

missons_file_str = "gym-minigrid/gym_minigrid/envs/missions/fetch_train_easy_test.json"

env = FetchAttrEnv(size=6,
                   numObjs=3,
                   missions_file_str=missons_file_str)

env = FrameStackerWrapper(LessActionAndObsWrapper(TextWrapper(env=env)))
# %%
device = 'cuda'

env = TorchWrapper(env, device=device)
model = BaseDoubleDQN(obs_space=env.observation_space,
                      action_space=env.action_space,
                      lr=3e-4,
                      device=device,
                      use_memory=True
                      )
model.to(device)

# %%

n_episodes = 10000
total_step = 1

# SweetLogger is a tensorboardXWriter with additionnal tool to help dealing with lists
tf_logger = SweetLogger(dump_step=1000, path_to_log="out_test/single")

# When do you want to store images of q-function and corresponding state ?
# Specify here :
q_values_generator = QValueVisualizer(proba_log=0.1)


print(env.observation_space)
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

        loss = model.optimize_model(state=obs["image"],
                                    action=act,
                                    next_state=new_obs["image"],
                                    reward=reward,
                                    done=done,
                                    mission=new_obs["mission"],
                                    environment_step=total_step)

        tf_logger.log("loss", loss)

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
    print("End of ep #{} Time since begin ep : {:.2f}, Time per step : {:.2f} Total iter : {}  iter this ep : {} rewrd : {:.3f}".format(
        episode_num, time_since_ep_start, time_since_ep_start / iter_this_ep, total_step, iter_this_ep, reward_this_ep))

    tf_logger.log("reward", reward_this_ep)
    tf_logger.log("epsilon", model.current_epsilon)






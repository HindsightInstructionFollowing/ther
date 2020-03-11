# %%

import collections
import random
import numpy as np
from algo.replay_buffer import ReplayBuffer, RecurrentReplayBuffer
np.random.seed(44)


test_normal_buffer = False
test_prioritize_buffer = False
test_prioritize_recurrent_buffer = True

# %%
if test_normal_buffer:

    config = {"hindsight_reward": 1,
              "size": 100,
              "use_her": True}

    buffer = ReplayBuffer(config)

    mission = ["nik tout"]
    action = 0
    reward = 0
    for elem in range(1,200):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 3 == 0:
            print("done")
            done = True
            hindsight_mission = [1,0]
            reward = 0

        buffer.add_transition(elem, action, reward, elem+1, done, mission, 2, hindsight_mission)
        print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
            len(buffer), buffer.position, len(buffer.current_episode), elem))
        print("buffer memory", buffer.memory)

    # %%

    print(buffer.sample(10))
    print(buffer.sample(10))

# %%

if test_prioritize_buffer :
    config = {
        "hindsight_reward" : 1,
        "use_her" : False,
        "size" : 10,
        "n_step" : 1,
        "gamma" : 0.99,
        "use_compression" : False,
        "prioritize": True,
        "prioritize_alpha" : 0.9,
        "prioritize_beta": 0.6,
        "prioritize_eps" : 1e-6,
    }

    buffer = ReplayBuffer(config=config)
    action = 0
    mission = ['lol']

    replayed_seq = dict()

    for elem in range(0,8):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 3 == 0:
            print("done")
            done = True
            hindsight_mission = [1,0]
            reward = 0

        buffer.add_transition(elem, action, reward, elem+1, done, mission, 2, hindsight_mission)
        print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
            len(buffer), buffer.position, len(buffer.current_episode), elem))
        print("buffer memory", buffer.memory)

    print(len(buffer))

    buffer.sample(3)
    print(buffer.last_id_sampled)
    error = np.array([2, 10, 2])
    buffer.update_transitions_proba(errors=error)

    print(buffer.prioritize_p)

    for i in range(100):
        sam, w = buffer.sample(3)

        for n, s in enumerate(sam):
            #print(s.current_state, w[n])
            if s.current_state in replayed_seq:
                replayed_seq[s.current_state]["count"] += 1
            else:
                replayed_seq[s.current_state] = dict()
                replayed_seq[s.current_state]["count"] = 1
                replayed_seq[s.current_state]["w"] = w[n]


    print(replayed_seq)



# %%

if test_prioritize_recurrent_buffer:

    config = {
        "hindsight_reward": 1,
        "use_her": False,
        "size": 10,
        "n_step": 1,
        "gamma": 0.99,
        "use_compression": False,
        "prioritize": True,
        "prioritize_alpha": 0.9,
        "prioritize_beta": 0.6,
        "prioritize_eps": 1e-6,
        "prioritize_max_mean_balance": 0.9,
    }

    buffer = RecurrentReplayBuffer(config=config)
    action = 0
    mission = ['lol']

    replayed_seq = dict()

    for elem in range(0, 8):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 3 == 0:
            print("done")
            done = True
            hindsight_mission = [1, 0]
            reward = 0

        buffer.add_transition(elem, action, reward, elem + 1, done, mission, 2, hindsight_mission)
        print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
            len(buffer), buffer.position, len(buffer.current_episode), elem))
        print("buffer memory", buffer.memory)

    print(len(buffer))

    s = buffer.sample(5)
    print(buffer.last_id_sampled)
    print(s)
    error = np.array([2, 2, 8, 10, 2, 2, 8])
    buffer.update_transitions_proba(errors=error)

    print(buffer.prioritize_p)

    for i in range(100):
        sam, w = buffer.sample(3)

        for n, s in enumerate(buffer.last_id_sampled):
            # print(s.current_state, w[n])
            seq_key = s
            if seq_key in replayed_seq:
                replayed_seq[seq_key]["count"] += 1
            else:
                replayed_seq[seq_key] = dict()
                replayed_seq[seq_key]["count"] = 1
                replayed_seq[seq_key]["w"] = w[n]

    print(replayed_seq)

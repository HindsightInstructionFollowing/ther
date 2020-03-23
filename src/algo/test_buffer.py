# %%
import pytest

import collections
import random
import torch
import numpy as np
from algo.replay_buffer_parallel import ReplayBufferParallel
from algo.learnt_her import LearntHindsightExperienceReplay

from replay_buffer_tools import MissionPadder

import json

class DummyActionSpace(object):
    def __init__(self, n=4):
        self.n = n

class DummyMission(object):
    def __init__(self, high):
        self.high = np.array([high])

class DummyEnv(object):
    def __init__(self, n_action=4, obs_space_shape=None):
        self.action_space = DummyActionSpace(n_action)
        if obs_space_shape is None:
            obs_space_shape = [1]

        mission = DummyMission(10)
        self.observation_space = {"image" : np.zeros(obs_space_shape),
                                  "mission" : mission}
        self.max_steps = 5
        self.mission_padder = MissionPadder(4, 0)

def test_normal_buffer_no_her():

    config = json.load(open("config/model/conv_fetch_minigrid.json", 'r'))
    config = config["algo_params"]["experience_replay_config"]

    config["use_her"] = False
    config["use_compression"] = False
    config["prioritize"] = False
    config["size"] = 10
    config["n_step"] = 1
    config["gamma"] = 0.99
    config["num_workers"] = 1
    config["batch_size"] = 2

    env = DummyEnv(4)

    buffer = ReplayBufferParallel(config, is_recurrent=False, env=env)

    mission = [42, 42]
    action = 0
    reward = 0
    for elem in range(1,15):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 3 == 0:
            # print("done")
            done = True
            hindsight_mission = [11, 11]
            reward = 0

        buffer.add_transition(elem, action, reward, elem+1, done, mission, 2, hindsight_mission)
        # print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
        #     len(buffer), buffer.dataset.position, len(buffer.current_episode), elem))
        # print("buffer memory", buffer.dataset.memory)

    assert buffer.dataset.memory[0].current_state == 10
    assert buffer.dataset.memory[0].next_state == 11
    assert buffer.dataset.memory[0].mission == [42,42]

    assert len(buffer) == 9
    assert buffer.dataset.position == 3
    assert buffer.batch_sampler.n_memory_cell == 9

    s, w = buffer.sample()
    assert w == 1
    assert buffer.batch_sampler.prioritize_p.sum() == 9
    assert buffer.batch_sampler.prioritize_proba.sum() == 1

    assert np.all(buffer.batch_sampler.prioritize_p == [1] * 9 + [0])
    assert np.all(buffer.batch_sampler.prioritize_proba == [1/9] * 9)

    buffer.update_transitions_proba(np.array([10, 10]))

    s, w = buffer.sample()
    assert w == 1
    assert np.all(buffer.batch_sampler.prioritize_p == [1] * 9 + [0])
    assert np.all(buffer.batch_sampler.prioritize_proba == [1 / 9] * 9)

def test_normal_buffer_her():

    config = json.load(open("config/model/conv_fetch_minigrid.json", 'r'))
    config = config["algo_params"]["experience_replay_config"]
    env = DummyEnv(4)

    config["use_her"] = True
    config["use_compression"] = False
    config["prioritze"] = False
    config["size"] = 10
    config["n_step"] = 1
    config["gamma"] = 0.99
    config["num_workers"] = 1

    buffer = ReplayBufferParallel(config, is_recurrent=False, env=env)

    mission = [42, 42]
    action = 0
    reward = 0
    for elem in range(1, 7):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 3 == 0:
            # print("done")
            done = True
            hindsight_mission = [11, 11]
            reward = 0

        buffer.add_transition(elem, action, reward, elem + 1, done, mission, 2, hindsight_mission)
        # print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
        #     len(buffer), buffer.dataset.position, len(buffer.current_episode), elem))
        # print("buffer memory", buffer.dataset.memory)
        if elem == 4:
            assert buffer.dataset.memory[0].current_state == 1
            assert buffer.dataset.memory[0].next_state == 2
            assert buffer.dataset.memory[0].mission == [11, 11]

            assert buffer.dataset.memory[3].current_state == 1
            assert buffer.dataset.memory[3].next_state == 2
            assert buffer.dataset.memory[3].mission == [42, 42]

            assert len(buffer) == 6
            assert buffer.dataset.position == 6

    assert buffer.dataset.memory[0].current_state == 4
    assert buffer.dataset.memory[0].next_state == 5
    assert buffer.dataset.memory[0].mission == [42, 42]

    assert len(buffer) == 9
    assert buffer.dataset.position == 3


def test_normal_buffer_her_nstep():
    config = json.load(open("config/model/conv_fetch_minigrid.json", 'r'))
    config = config["algo_params"]["experience_replay_config"]
    env = DummyEnv(4)

    config["use_her"] = True
    config["use_compression"] = False
    config["prioritze"] = False
    config["size"] = 10
    config["n_step"] = 3
    config["gamma"] = 0.99
    config["num_workers"] = 1

    buffer = ReplayBufferParallel(config, is_recurrent=False, env=env, logger=None, device='cpu')

    mission = [42, 42]
    action = 0
    reward = 0
    for elem in range(1, 9):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 4 == 0:
            # print("done")
            done = True
            hindsight_mission = [11, 11]
            reward = 0

        buffer.add_transition(elem, action, reward, elem + 1, done, mission, 2, hindsight_mission)
        # print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
        #     len(buffer), buffer.dataset.position, len(buffer.current_episode), elem))
        # print("buffer memory", buffer.dataset.memory)
        if elem == 4:
            assert buffer.dataset.memory[0].current_state == 1
            assert buffer.dataset.memory[0].next_state == 4
            assert buffer.dataset.memory[0].mission == [11,11]
            assert buffer.dataset.memory[0].terminal == False
            assert buffer.dataset.memory[0].reward == 0
            assert buffer.dataset.memory[0].gamma == 0.99 ** 3

            assert buffer.dataset.memory[1].current_state == 2
            assert buffer.dataset.memory[1].next_state == 5
            assert buffer.dataset.memory[1].mission == [11, 11]
            assert buffer.dataset.memory[1].reward == 0.99 ** 2
            assert buffer.dataset.memory[1].gamma == 0.99 ** 3

            assert buffer.dataset.memory[2].current_state == 3
            assert buffer.dataset.memory[2].next_state == 5
            assert buffer.dataset.memory[2].mission == [11, 11]
            assert buffer.dataset.memory[2].terminal == True
            assert buffer.dataset.memory[2].reward == 0.99
            assert buffer.dataset.memory[2].gamma == 0.99 ** 2

            assert buffer.dataset.memory[3].current_state == 4
            assert buffer.dataset.memory[3].next_state == 5
            assert buffer.dataset.memory[3].mission == [11, 11]
            assert buffer.dataset.memory[3].reward == 1
            assert buffer.dataset.memory[3].terminal == True
            assert buffer.dataset.memory[3].gamma == 0.99

            assert buffer.dataset.memory[4].current_state == 1
            assert buffer.dataset.memory[4].next_state == 4
            assert buffer.dataset.memory[4].mission == [42, 42]
            assert buffer.dataset.memory[4].reward == 0
            assert buffer.dataset.memory[4].gamma == 0.99 ** 3

            assert buffer.dataset.memory[5].current_state == 2
            assert buffer.dataset.memory[5].next_state == 5
            assert buffer.dataset.memory[5].mission == [42, 42]
            assert buffer.dataset.memory[5].reward == 0
            assert buffer.dataset.memory[5].gamma == 0.99 ** 3

            assert len(buffer) == 8
            assert buffer.dataset.position == 8

    assert buffer.dataset.memory[0].current_state == 5
    assert buffer.dataset.memory[0].next_state == 8
    assert buffer.dataset.memory[0].mission == [11, 11]
    assert buffer.dataset.memory[0].reward == 0
    assert buffer.dataset.memory[0].terminal == False
    assert buffer.dataset.memory[0].gamma == 0.99 ** 3

    assert buffer.dataset.memory[1].current_state == 6
    assert buffer.dataset.memory[1].next_state == 9
    assert buffer.dataset.memory[1].mission == [11, 11]
    assert buffer.dataset.memory[1].reward == 0.99 ** 2
    assert buffer.dataset.memory[1].terminal == True
    assert buffer.dataset.memory[1].gamma == 0.99 ** 3

    assert len(buffer) == 8
    assert buffer.dataset.position == 8

def test_normal_buffer_her_prio_nstep():

    config = json.load(open("config/model/conv_fetch_minigrid.json", 'r'))
    config = config["algo_params"]["experience_replay_config"]
    env = DummyEnv(4)

    config["use_her"] = True
    config["use_compression"] = False
    config["prioritze"] = True

    config["size"] = 10
    config["n_step"] = 3
    config["gamma"] = 0.99
    config["num_workers"] = 1
    config["batch_size"] = 2


    buffer = ReplayBufferParallel(config, is_recurrent=False, env=env)

    mission = [42, 42]
    action = 0
    reward = 0
    for elem in range(1, 9):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 4 == 0:
            # print("done")
            done = True
            hindsight_mission = [11, 11]
            reward = 0

        buffer.add_transition(elem, action, reward, elem + 1, done, mission, 2, hindsight_mission)
        # print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
        #     len(buffer), buffer.dataset.position, len(buffer.current_episode), elem))
        # print("buffer memory", buffer.dataset.memory)
        if elem == 4:

            assert buffer.dataset.memory[5].current_state == 2
            assert buffer.dataset.memory[5].next_state == 5
            assert buffer.dataset.memory[5].mission == [42, 42]
            assert buffer.dataset.memory[5].reward == 0
            assert buffer.dataset.memory[5].gamma == 0.99 ** 3

            assert len(buffer) == 8
            assert buffer.dataset.position == 8

    assert buffer.dataset.memory[1].current_state == 6
    assert buffer.dataset.memory[1].next_state == 9
    assert buffer.dataset.memory[1].mission == [11, 11]
    assert buffer.dataset.memory[1].reward == 0.99 ** 2
    assert buffer.dataset.memory[1].terminal == True
    assert buffer.dataset.memory[1].gamma == 0.99 ** 3

    assert len(buffer) == 8
    assert buffer.dataset.position == 8

    s, w = buffer.sample()
    id_updated = buffer.batch_sampler.last_id_sampled

    assert np.all(w == [1,1])
    assert buffer.batch_sampler.prioritize_p.sum() == 8
    assert buffer.batch_sampler.prioritize_proba.sum() == 1

    assert np.all(buffer.batch_sampler.prioritize_p == [1] * 8 + [0, 0])
    assert np.all(buffer.batch_sampler.prioritize_proba == [1 / 8] * 8)

    buffer.update_transitions_proba(np.array([10, 10]))
    s, w = buffer.sample()

    id_updated2 = buffer.batch_sampler.last_id_sampled

    p = np.ones(10)
    p[id_updated] = 10 ** 0.9
    p[-2:] = 0
    proba = p[:8] / p.sum()

    assert np.isclose(buffer.batch_sampler.prioritize_p, p).all()
    assert np.isclose(buffer.batch_sampler.prioritize_proba, proba).all()

    buffer.update_transitions_proba(np.array([3, 3]))
    buffer.sample()

    p = np.ones(10)
    p[id_updated] = 10 ** 0.9
    p[id_updated2] = 3 ** 0.9
    p[-2:] = 0
    proba = p[:8] / p.sum()

    assert np.isclose(buffer.batch_sampler.prioritize_p, p).all()
    assert np.isclose(buffer.batch_sampler.prioritize_proba, proba).all()

    for elem in range(1, 9):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 4 == 0:
            # print("done")
            done = True
            hindsight_mission = [11, 11]
            reward = 0

        buffer.add_transition(elem, action, reward, elem + 1, done, mission, 2, hindsight_mission)
        if elem == 4:

            assert buffer.dataset.memory[5].current_state == 2
            assert buffer.dataset.memory[5].next_state == 5
            assert buffer.dataset.memory[5].mission == [42, 42]
            assert buffer.dataset.memory[5].reward == 0
            assert buffer.dataset.memory[5].gamma == 0.99 ** 3

            assert len(buffer) == 8
            assert buffer.dataset.position == 8

    s, w = buffer.sample()
    assert np.all(w == [1, 1])
    assert buffer.batch_sampler.prioritize_proba.sum() == 1
    assert np.all(buffer.batch_sampler.prioritize_proba == [1 / 8] * 8)


def test_recurrent_buffer():

    config = json.load(open("config/model/conv_vizdoom_recurrent.json", 'r'))
    config = config["algo_params"]["experience_replay_config"]

    config["use_her"] = True
    config["use_compression"] = False
    config["prioritize"] = True
    config["size"] = 10
    config["batch_size"] = 2
    config["n_step"] = 1
    config["gamma"] = 0.99
    config["num_workers"] = 1
    env = DummyEnv(4)

    buffer = ReplayBufferParallel(config, is_recurrent=True, env=env, recurrent_memory_saving=2)
    action = 0
    mission = [42, 42]

    for elem in range(1, 9):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 3 == 0:
            done = True
            hindsight_mission = [11, 11]
            reward = 0

        buffer.add_transition(np.array([[elem]]), action, reward, np.array([[elem + 1]]), done, mission, np.array([2]), hindsight_mission)
        # print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
        #     len(buffer), buffer.position, len(buffer.current_episode), elem))
        # print("buffer memory", buffer.memory)

        if elem == 4:
            assert buffer.dataset.n_memory_cell == 2
            assert buffer.dataset.memory[0][0].current_state == 1
            assert buffer.dataset.memory[0][0].next_state == 2
            assert buffer.dataset.memory[0][0].terminal == False
            assert buffer.dataset.memory[0][0].reward == 0
            assert buffer.dataset.memory[0][0].mission == [11, 11]
            assert buffer.dataset.memory[0][0].gamma == 0.99

            assert buffer.dataset.memory[0][1].current_state == 2
            assert buffer.dataset.memory[0][1].next_state == 3
            assert buffer.dataset.memory[0][1].terminal == False
            assert buffer.dataset.memory[0][1].reward == 0
            assert buffer.dataset.memory[0][1].mission == [11, 11]
            assert buffer.dataset.memory[0][1].gamma == 0.99

            assert buffer.dataset.memory[0][2].current_state == 3
            assert buffer.dataset.memory[0][2].next_state == 4
            assert buffer.dataset.memory[0][2].terminal == True
            assert buffer.dataset.memory[0][2].reward == 1
            assert buffer.dataset.memory[0][2].mission == [11, 11]
            assert buffer.dataset.memory[0][2].gamma == 0.99

            assert buffer.dataset.memory[1][0].current_state == 1
            assert buffer.dataset.memory[1][0].next_state == 2
            assert buffer.dataset.memory[1][0].terminal == False
            assert buffer.dataset.memory[1][0].mission == [42, 42]
            assert buffer.dataset.memory[1][0].reward == 0
            assert buffer.dataset.memory[1][0].gamma == 0.99

            assert buffer.dataset.memory[1][2].current_state == 3
            assert buffer.dataset.memory[1][2].next_state == 4
            assert buffer.dataset.memory[1][2].terminal == True
            assert buffer.dataset.memory[1][2].reward == 0
            assert buffer.dataset.memory[1][2].mission == [42, 42]
            assert buffer.dataset.memory[1][2].gamma == 0.99


    s, w = buffer.sample()
    id_sampled = buffer.batch_sampler.last_id_sampled
    assert np.all(w == [1,1])
    assert buffer.batch_sampler.prioritize_proba.sum() == 1
    assert np.all(buffer.batch_sampler.prioritize_proba == [1 / 4] * 4)
    assert len(buffer) == 4

    buffer.update_transitions_proba([5,7.5,10,5,7.5,10])
    p = np.ones(5)
    p[4] = 0
    p[id_sampled] = (10 * 0.9 + 7.5 * 0.1) ** 0.9

    assert np.isclose(p, buffer.batch_sampler.prioritize_p, atol=0.1).all()


def test_recurrent_buffer_her_nstep():
    config = json.load(open("config/model/conv_vizdoom_recurrent.json", 'r'))
    config = config["algo_params"]["experience_replay_config"]

    config["use_her"] = True
    config["use_compression"] = False
    config["prioritize"] = True
    config["size"] = 10
    config["batch_size"] = 2
    config["n_step"] = 3
    config["gamma"] = 0.99
    config["num_workers"] = 1
    env = DummyEnv(4)

    buffer = ReplayBufferParallel(config, is_recurrent=True, env=env, recurrent_memory_saving=2)
    action = 0
    mission = [42, 42]

    for elem in range(1, 9):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 4 == 0:
            done = True
            hindsight_mission = [11, 11]
            reward = 0

        buffer.add_transition(np.array([[elem]]), action, reward, np.array([[elem + 1]]), done, mission, np.array([2]), hindsight_mission)
        # print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
        #     len(buffer), buffer.position, len(buffer.current_episode), elem))
        # print("buffer memory", buffer.memory)

        if elem == 4:

            print(buffer.dataset.memory)
            assert buffer.dataset.n_memory_cell == 2
            assert buffer.dataset.memory[0][0].current_state == 1
            assert buffer.dataset.memory[0][0].next_state == 4
            assert buffer.dataset.memory[0][0].terminal == False
            assert buffer.dataset.memory[0][0].reward == 0
            assert buffer.dataset.memory[0][0].mission == [11, 11]
            assert buffer.dataset.memory[0][0].gamma == 0.99 ** 3

            assert buffer.dataset.memory[0][1].current_state == 2
            assert buffer.dataset.memory[0][1].next_state == 5
            assert buffer.dataset.memory[0][1].terminal == True
            assert buffer.dataset.memory[0][1].reward == 0.99 ** 2
            assert buffer.dataset.memory[0][1].mission == [11, 11]
            assert buffer.dataset.memory[0][1].gamma == 0.99 ** 3

            assert buffer.dataset.memory[0][2].current_state == 3
            assert buffer.dataset.memory[0][2].next_state == 5
            assert buffer.dataset.memory[0][2].terminal == True
            assert buffer.dataset.memory[0][2].reward == 0.99
            assert buffer.dataset.memory[0][2].mission == [11, 11]
            assert buffer.dataset.memory[0][2].gamma == 0.99 ** 2

            assert buffer.dataset.memory[0][3].current_state == 4
            assert buffer.dataset.memory[0][3].next_state == 5
            assert buffer.dataset.memory[0][3].terminal == True
            assert buffer.dataset.memory[0][3].reward == 1
            assert buffer.dataset.memory[0][3].mission == [11, 11]
            assert buffer.dataset.memory[0][3].gamma == 0.99

            assert buffer.dataset.memory[1][0].current_state == 1
            assert buffer.dataset.memory[1][0].next_state == 4
            assert buffer.dataset.memory[1][0].terminal == False
            assert buffer.dataset.memory[1][0].mission == [42, 42]
            assert buffer.dataset.memory[1][0].reward == 0
            assert buffer.dataset.memory[1][0].gamma == 0.99 ** 3

            assert buffer.dataset.memory[1][2].current_state == 3
            assert buffer.dataset.memory[1][2].next_state == 5
            assert buffer.dataset.memory[1][2].terminal == True
            assert buffer.dataset.memory[1][2].reward == 0
            assert buffer.dataset.memory[1][2].mission == [42, 42]
            assert buffer.dataset.memory[1][2].gamma == 0.99 ** 2


    s, w = buffer.sample()
    id_sampled = buffer.batch_sampler.last_id_sampled
    assert np.all(w == [1,1])
    assert buffer.batch_sampler.prioritize_proba.sum() == 1
    assert np.all(buffer.batch_sampler.prioritize_proba == [1 / 3] * 3)
    assert len(buffer) == 3
    print(buffer.dataset.memory)

    buffer.update_transitions_proba([5,7.5,10,5,7.5,10])
    p = np.ones(5)
    p[4] = 0
    p[3] = 0

    p[id_sampled] = (10 * 0.9 + 7.5 * 0.1) ** 0.9

    assert np.isclose(p, buffer.batch_sampler.prioritize_p, atol=0.1).all()


def test_normal_ther():

    config = json.load(open("config/model/conv_fetch_minigrid.json", 'r'))

    config = config["algo_params"]["experience_replay_config"]

    config["use_ther"] = True
    config["use_compression"] = False
    config["prioritize"] = True
    config["size"] = 10
    config["batch_size"] = 2
    config["n_step"] = 3
    config["gamma"] = 0.99
    config["num_workers"] = 1

    obs_shape = (1,7,7)
    env = DummyEnv(4, obs_shape)

    buffer = LearntHindsightExperienceReplay(config, is_recurrent=False, env=env, recurrent_memory_saving=2)
    action = 0
    mission = [42, 42]

    for elem in range(1, 13):
        done = False
        hindsight_mission = None
        reward = 0
        if elem % 4 == 0:
            done = True
            hindsight_mission = [11, 11]
            reward = 0

        e = np.empty(obs_shape)
        e[:] = elem

        e_plus1 = np.empty(obs_shape)
        e_plus1[:] = elem+1
        buffer.add_transition(e, action, reward, e_plus1, done, mission, np.array([2]),
                              hindsight_mission)
        # print("len : {}, Position : {}, current_episode_size_in_buffer {}, current_step {}".format(
        #     len(buffer), buffer.position, len(buffer.current_episode), elem))
        # print("buffer memory", buffer.memory)

        if elem == 4:
            assert buffer.dataset.memory[0].current_state.max() == 1
            assert buffer.dataset.memory[0].next_state.max() == 4
            assert buffer.dataset.memory[0].mission == [42, 42]
            assert buffer.dataset.memory[0].reward == 0
            assert buffer.dataset.memory[0].gamma == 0.99 ** 3

            assert buffer.dataset.memory[1].current_state.max() == 2
            assert buffer.dataset.memory[1].next_state.max() == 5
            assert buffer.dataset.memory[1].mission == [42, 42]
            assert buffer.dataset.memory[1].reward == 0
            assert buffer.dataset.memory[1].gamma == 0.99 ** 3

            assert len(buffer) == 4
            assert buffer.dataset.position == 4

    print(buffer.dataset.memory)
    assert buffer.dataset.memory[0].current_state.max() == 9
    assert buffer.dataset.memory[0].next_state.max() == 12
    assert buffer.dataset.memory[0].mission == [42, 42]
    assert buffer.dataset.memory[0].reward == 0
    assert buffer.dataset.memory[0].terminal == False
    assert buffer.dataset.memory[0].gamma == 0.99 ** 3

    assert buffer.dataset.memory[1].current_state.max() == 10
    assert buffer.dataset.memory[1].next_state.max() == 13
    assert buffer.dataset.memory[1].mission == [42, 42]
    assert buffer.dataset.memory[1].reward == 0
    assert buffer.dataset.memory[1].terminal == True
    assert buffer.dataset.memory[1].gamma == 0.99 ** 3

    s, w = buffer.sample()
    id_sampled = buffer.batch_sampler.last_id_sampled
    assert np.all(w == [1, 1])
    assert buffer.batch_sampler.prioritize_proba.sum() == 1
    assert np.all(buffer.batch_sampler.prioritize_proba == [1 / 8] * 8)
    assert len(buffer) == 8

    buffer.update_transitions_proba(np.array([10, 10]))
    p = np.ones(10)
    p[9] = 0
    p[8] = 0

    p[id_sampled] = 10 ** 0.9
    assert np.isclose(p, buffer.batch_sampler.prioritize_p, atol=0.1).all()


# test_normal_buffer_her()
# test_normal_buffer_no_her()
# test_normal_buffer_her_nstep()
# test_normal_buffer_her_prio_nstep()
#
# test_recurrent_buffer()
# test_recurrent_buffer_her_nstep()
# test_normal_ther()
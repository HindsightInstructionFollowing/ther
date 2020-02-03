import pickle as pkl
import random
from algo.learnt_her import LearntHindsightRecurrentExperienceReplay
import torch

config = {
    "hindsight_reward": 0.8,
    "use_her": False,
    "size": 40000,
    "n_step" : 1,
    "gamma" : 0.99,

    "use_compression" : False,

    "ther_params": {
        "max_steps_optim" : 2000,
        "lr": 3e-4,
        "batch_size": 64,
        "weight_decay": 1e-4,
        "update_steps": [30, 300, 1000],
        "n_sample_before_using_generator": 300,
        "tolerance_convergence": 1e-5,

        "architecture_params": {
            "conv_layers_channel": [32, 64, 128],
            "conv_layers_size": [2, 2, 2],
            "conv_layers_stride": [1, 1, 1],
            "max_pool_layers": [2, 0, 0],
            "embedding_dim": 32,
            "generator_max_len": 10,

            "dropout": 0.2,
            "decoder_hidden" : 128
        }
    }}

device = 'cuda'

import json
vocab = json.load(open("gym-minigrid/gym_minigrid/envs/missions/vocab_fetch.json", "r"))
i2w = list(vocab["vocabulary"].keys())

input_shape = (16, 7, 7)
n_output = 27

full_dataset = pkl.load(open("generator_collected_dataset.pkl", "rb"))
n_sample = int(0.80 * len(full_dataset["states"]))


replay = LearntHindsightRecurrentExperienceReplay(input_shape, n_output, config, device)

replay.generator_dataset = {
    "states" : full_dataset["states"][:n_sample],
    "instructions" : full_dataset["instructions"][:n_sample],
    "lengths" : full_dataset["lengths"][:n_sample]
}


while True:

    replay.instruction_generator.train()
    replay._train_generator()
    predicted_list = []
    replay.instruction_generator.train(mode=False)

    for test_state, test_instruction, length in \
            zip(full_dataset["states"][n_sample:], full_dataset["instructions"][n_sample:], full_dataset["lengths"][n_sample:]):

        test_state = torch.Tensor(test_state).to(device)

        generated_instruction = replay.instruction_generator.generate(test_state, test_instruction, length)
        predicted_list.append(replay._cheat_check(true_mission=test_instruction, generated_mission=generated_instruction))
    print("Mean cheat check", sum(predicted_list) / len(predicted_list))

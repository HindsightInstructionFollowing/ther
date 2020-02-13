import pickle as pkl
import random
from algo.learnt_her import LearntHindsightRecurrentExperienceReplay
import torch

from nltk.translate import bleu_score
import numpy as np
import pickle as pkl

def accuracy(prediction_sentence, true_sentence):
    min_length = min(prediction_sentence.size(0), true_sentence.size(0))

    accuracy = 0
    for id in range(min_length):
        accuracy += int(prediction_sentence[id] == true_sentence[id])

    return accuracy / min_length

config = {
    "hindsight_reward": 0.8,
    "use_her": False,
    "size": 40000,
    "n_step" : 1,
    "gamma" : 0.99,

    "use_compression" : False,

    "ther_params": {
        "accuracy_convergence": 1,
        "max_steps_optim" : 2000,
        "tolerance_convergence": 1e-5,

        "lr": 1e-4,
        "batch_size": 64,
        "weight_decay": 0,
        "update_steps": [30, 300, 1000],
        "n_sample_before_using_generator": 300,

        "n_state_to_predict_instruction": 5,

        "architecture_params": {
            "conv_layers_channel" : [32, 64, 64],
            "conv_layers_size" : [8,4,4],
            "conv_layers_stride" : [4,2,2],
            "max_pool_layers": [0, 0, 0],

            "projection_after_conv": 512,

            "trajectory_encoding_rnn": 512,

            "embedding_dim": 32,
            "generator_max_len": 10,

            "dropout": 0,
            "decoder_hidden" : 512
        }
    }}

device = 'cuda'

import json
# vocab = json.load(open("gym-minigrid/gym_minigrid/envs/missions/vocab_fetch.json", "r"))
vocab = pkl.load(open("vocabulary_vizdoom.pkl", "rb"))
i2w = list(vocab.keys())

input_shape = (3, 84, 84)
n_output = 23

full_dataset = pkl.load(open("generator_dataset7.pkl", "rb"))
n_sample = int(0.80 * len(full_dataset["states"]))

print("Number of samples : ", n_sample)

replay = LearntHindsightRecurrentExperienceReplay(input_shape, n_output, config, device)

replay.generator_dataset = {
    "states" : full_dataset["states"][:n_sample],
    "instructions" : full_dataset["instructions"][:n_sample],
    "lengths" : full_dataset["lengths"][:n_sample]
}


while True:

    replay._train_generator()
    bleu2 = []
    bleu1 = []
    accs = []

    for test_state, test_instruction, length in \
            zip(full_dataset["states"][n_sample:], full_dataset["instructions"][n_sample:], full_dataset["lengths"][n_sample:]):

        test_state = torch.Tensor(test_state).squeeze(0).to(device)

        generated_instruction = replay.instruction_generator.generate(test_state)
        #predicted_list.append(replay._cheat_check(true_mission=test_instruction, generated_mission=generated_instruction))
        bleu2.append(bleu_score.sentence_bleu(references=[test_instruction],
                                              hypothesis=generated_instruction,
                                              smoothing_function=bleu_score.SmoothingFunction().method2,
                                              weights=(0.5, 0.5)))

        bleu1.append(bleu_score.sentence_bleu(references=[test_instruction],
                                              hypothesis=generated_instruction,
                                              smoothing_function=bleu_score.SmoothingFunction().method2,
                                              weights=[1]))

        accs.append(accuracy(generated_instruction, test_instruction))


    #print("Mean cheat check", sum(predicted_list) / len(predicted_list))
    print("Mean bleu2 ", np.mean(bleu2))
    print("Mean bleu1 ", np.mean(bleu1))
    print("Mean acc test ", np.mean(accs))



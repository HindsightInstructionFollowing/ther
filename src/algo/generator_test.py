import pickle as pkl
import random
from algo.learnt_her import LearntHindsightRecurrentExperienceReplay
import torch

from nltk.translate import bleu_score
import numpy as np
import pickle as pkl
import json
from config import override_config_recurs
import matplotlib.pyplot as plt


def translate(sentence, i2w):
    for id in sentence:
        print(i2w[id], end=' ')
    print("")

def save_images(test_state):
    for im in range(test_state.size(0)):
        i = test_state[im].permute(1, 2, 0).cpu().numpy() / 255
        plt.imsave("state{}.png".format(im), i)

def post_process(obj_name, test_instruction, generated_instruction, test_state, i2w):
    print(obj_name)
    translate(test_instruction, i2w)
    translate(generated_instruction, i2w)
    save_images(test_state)

def accuracy(prediction_sentence, true_sentence):
    min_length = min(prediction_sentence.shape[0], true_sentence.shape[0])

    accuracy = 0
    for id in range(min_length):
        accuracy += int(prediction_sentence[id] == true_sentence[id])

    return accuracy / min_length

ext_config = {
    "ther_params": {
        "accuracy_convergence": 0.96,
        "max_steps_optim" : 400,
        "tolerance_convergence": 1e-5,

        "lr": 3e-4,
        "batch_size": 128,
        "weight_decay": 1e-3,
        "update_steps": [30, 300, 1000],
        "n_sample_before_using_generator": 300,

        "n_state_to_predict_instruction": 7,

        "architecture_params": {
            "conv_layers_channel" : [32, 64, 64],
            "conv_layers_size" : [8,4,4],
            "conv_layers_stride" : [4,2,2],
            "max_pool_layers": [0, 0, 0],

            "projection_after_conv": 512,

            "trajectory_encoding_rnn": 512,

            "embedding_dim": 512,
            "generator_max_len": 10,

            "dropout": 0.8,
            "decoder_hidden": 512
        }
    }}



config = json.load(open("config/model/conv_vizdoom_recurrent.json"))

n_step = config["algo_params"]["n_step"]
gamma = config["algo_params"]["gamma"]

config = config["algo_params"]["experience_replay_config"]
config = override_config_recurs(config, ext_config)

config["n_step"] = n_step
config["gamma"] = gamma

device = 'cuda'

import json
# vocab = json.load(open("gym-minigrid/gym_minigrid/envs/missions/vocab_fetch.json", "r"))
all_instructions = pkl.load(open("saved_tools/obj2instructions.pkl","rb"))
vocab = pkl.load(open("saved_tools/vocabulary_vizdoom.pkl", "rb"))
i2w = list(vocab.keys())


full_dataset = pkl.load(open("generator_dataset7.pkl", "rb"))
n_sample = int(0.8 * len(full_dataset["states"]))

input_shape = tuple(full_dataset["states"][0].size())
n_output = 23

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

    bleu2_cheat = []
    bleu1_cheat = []

    accs = []

    for test_state, test_instruction, length, obj_name in \
            zip(full_dataset["states"][n_sample:], full_dataset["instructions"][n_sample:], full_dataset["lengths"][n_sample:], full_dataset["correct_obj_name"][n_sample:]):

        test_state = torch.Tensor(test_state).squeeze(0).to(device)

        generated_instruction = replay.generate(test_state)

        all_references = [[vocab[w] for w in sentence.split()] for sentence in all_instructions[obj_name]]
        test_instruction = test_instruction[1:-1].numpy()
        generated_instruction = generated_instruction.numpy()

        #predicted_list.append(replay._cheat_check(true_mission=test_instruction, generated_mission=generated_instruction))
        bleu2.append(bleu_score.sentence_bleu(references=[test_instruction],
                                              hypothesis=generated_instruction,
                                              smoothing_function=bleu_score.SmoothingFunction().method2,
                                              weights=(0.5, 0.5)))

        bleu1.append(bleu_score.sentence_bleu(references=[test_instruction],
                                              hypothesis=generated_instruction,
                                              smoothing_function=bleu_score.SmoothingFunction().method2,
                                              weights=[1]))

        bleu2_cheat.append(bleu_score.sentence_bleu(references=all_references,
                                              hypothesis=generated_instruction,
                                              smoothing_function=bleu_score.SmoothingFunction().method2,
                                              weights=(0.5, 0.5)))

        bleu1_cheat.append(bleu_score.sentence_bleu(references=all_references,
                                              hypothesis=generated_instruction,
                                              smoothing_function=bleu_score.SmoothingFunction().method2,
                                              weights=[1]))

        accs.append(accuracy(generated_instruction, test_instruction))


    #print("Mean cheat check", sum(predicted_list) / len(predicted_list))
    print("Mean bleu2 ", np.mean(bleu2))
    print("Mean bleu1 ", np.mean(bleu1))
    print("Mean bleu2_cheat ", np.mean(bleu2_cheat))
    print("Mean bleu1_cheat ", np.mean(bleu1_cheat))
    print("Mean acc test ", np.mean(accs))



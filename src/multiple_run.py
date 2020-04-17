import matplotlib
matplotlib.use('Agg')

from main import start_experiment
import argparse

import time

import json
import ray

from random import shuffle

from config import read_multiple_ext_file, read_multiple_config_file, extend_multiple_seed

if __name__ == "__main__":

    parser = argparse.ArgumentParser('Log Parser arguments!')

    parser.add_argument("-multiple_ext_config", type=str)
    parser.add_argument("-multiple_run_config", type=str)

    parser.add_argument("-n_gpus", type=int, default=4)
    parser.add_argument("-n_seeds", type=int, default=1)

    parser.add_argument("-out_dir", type=str)


    args = parser.parse_args()

    configs = []
    if args.multiple_run_config:
        configs.extend(read_multiple_config_file(args.multiple_run_config))

    if args.multiple_ext_config:
        configs.extend(read_multiple_ext_file(args.multiple_ext_config))

    if args.n_seeds > 1:
        configs = extend_multiple_seed(configs, number_of_seed=args.n_seeds)

    ray.init(num_gpus=args.n_gpus)

    print("Number of expe to launch : {}".format(len(configs)))

    #shuffle(configs)

    if args.out_dir:
        for config in configs:
            config["exp_dir"] = args.out_dir

    done = False
    list_success = ray.get([start_experiment.remote(**config) for config in configs])

    print("All expes done, great !")
    #ray.get([train.remote(**config) for config in configs[10:12]])

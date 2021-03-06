import json
import os
import sys
import itertools as it

from copy import deepcopy
from random import shuffle, randint

EXPE_DEFAULT_CONFIG = {
    "model_ext": '',
    "seed": 42,
    "exp_dir": "default_out",
    "local_test" : False
}

def override_config_recurs(config, config_extension):
    for key, value in config_extension.items():
        if type(value) is dict:
            config[key] = override_config_recurs(config[key], config_extension[key])
        else:
            assert key in config, "Warning, key defined in extension but not original : new key is {}".format(key)

            # Don't override names, change add name extension to the original
            if key == "name":
                config["name"] = config["name"] + "_" + value
            else:
                config[key] = value

    return config


def load_single_config(config_path):
    return json.load(open(config_path, "r"))

def check_json_intregrity(config_file_path, config_dict):
    config_file = open(config_file_path, 'r')
    config_dict_loaded = json.load(config_file)

    # if config_dict_loaded != config_dict:
    #     print("Warning, config on disk and specified by params are different")
    assert config_dict_loaded == config_dict, \
        """
        Error in config file handling, config_file on disk and this one must be the same !"
        config_dict :        {}
        ========
        config_dict_loaded : {}

        """.format(config_dict, config_dict_loaded)


def load_config(env_config_file, model_config_file, seed,
                out_dir,
                env_ext_file=None,
                model_ext_file=None
                ):

    print(env_config_file)
    print(model_config_file)
    print(model_ext_file)

    # === Loading ENV config, extension and check integrity =====
    # ===========================================================
    if type(env_config_file) is str:
        env_config = load_single_config(os.path.join("config", "env", env_config_file))
    else:
        assert type(env_config_file) is dict, \
            "Can be dict or str, but not something else, is {}\n{}".format(type(env_config_file), env_config_file)
        env_config = env_config_file

    # Override env file if specified
    if env_ext_file:
        env_ext_config = load_single_config(os.path.join("config", "env_ext", env_ext_file))
        env_config = override_config_recurs(env_config, env_ext_config)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # create env_file if necessary
    env_name = env_config["name"]
    env_path = os.path.join(out_dir, env_name)

    if not os.path.exists(env_path):
        os.mkdir(env_path)

    env_config_path = os.path.join(env_path, "env_config.json")
    if not os.path.exists(env_config_path):
        config_file = open(env_config_path, 'w')
        json.dump(obj=env_config, fp=config_file)
    else:
        if "osef" not in env_path: # Osef is the test setting, to avoid manually deleting
            check_json_intregrity(config_file_path=env_config_path,
                                  config_dict=env_config)

    # === Loading MODEL config, extension and check integrity =====
    # ===========================================================
    if type(model_config_file) is str:
        model_config = load_single_config(os.path.join("config", "model", model_config_file))
    else:
        assert type(model_config_file) is dict, "Problem, should be dict is {}\n{}".format(type(model_config_file),
                                                                                           model_config_file)
        model_config = model_config_file

    # Override model file if specified
    # Can be a dict of parameters or a str indicating the path to the extension
    if model_ext_file:
        if type(model_ext_file) is str:
            model_ext_config = load_single_config(os.path.join("config", "model_ext", model_ext_file))
        else:
            assert type(model_ext_file) is dict, "Not a dict problem, type : {}".format(type(model_ext_file))
            model_ext_config = model_ext_file

        model_config = override_config_recurs(model_config, model_ext_config)
    else:
        model_ext_config = {"name": ''}

    # create model_file if necessary
    model_name = model_config["name"]
    model_path = os.path.join(env_path, model_name)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model_config_path = os.path.join(model_path, "model_full_config.json")
    if not os.path.exists(model_config_path):
        config_file = open(model_config_path, 'w')
        json.dump(obj=model_config, fp=config_file, indent='    ', separators=(',',':'))

        # Dump the extension file too, easier to visualize quickly
        model_ext_config_path = os.path.join(model_path, "model_ext_config.json")
        model_ext_config_file = open(model_ext_config_path, 'w')
        json.dump(obj=model_ext_config, fp=model_ext_config_file, indent='    ', separators=(',',':'))

    else:
        if "osef" not in model_path:
            check_json_intregrity(config_file_path=model_config_path,
                                  config_dict=model_config)

    # Merge env and model config into one dict
    full_config = {**model_config, **env_config}
    full_config["model_name"] = model_config["name"]
    full_config["env_name"] = env_config["name"]
    del full_config["name"]

    # set seed
    full_config["seed"] = seed
    set_seed(seed)
    path_to_expe = os.path.join(model_path, str(seed))

    if not os.path.exists(path_to_expe):
        os.mkdir(path_to_expe)

    print(full_config["model_name"])
    print(full_config["env_name"])
    return full_config, path_to_expe


def read_multiple_ext_file(config_path):
    json_config = json.load(open(os.path.join("config/multiple_run_config", config_path), "r"))

    all_expe_to_run = []

    for ext in json_config["model_ext"]:
        expe_config = deepcopy(EXPE_DEFAULT_CONFIG)
        expe_config.update(json_config["common"])

        expe_config["model_ext"] = ext

        all_expe_to_run.append(expe_config)

    env_ext_run = []

    if json_config.get("env_ext", False):
        for env_ext in json_config["env_ext"]:

            for expe in all_expe_to_run:
                temp_expe = deepcopy(expe)
                temp_expe["env_ext"] = env_ext

                env_ext_run.append(temp_expe)

    all_expe_to_run.extend(env_ext_run)

    return all_expe_to_run

def read_multiple_config_file(config_path):
    json_config = json.load(open(os.path.join("config/multiple_run_config", config_path), "r"))
    assert type(json_config) == list, "Should be a list"

    all_expe_to_run = []

    for config in json_config:
        expe_config = deepcopy(EXPE_DEFAULT_CONFIG)
        expe_config.update(config)
        all_expe_to_run.append(expe_config)

    return all_expe_to_run

def extend_multiple_seed(all_expe_to_run, number_of_seed=2):
    extended_expe_list = []
    for expe in all_expe_to_run:
        for n_seed in range(number_of_seed):
            new_expe = deepcopy(expe)
            new_expe["seed"] = n_seed
            extended_expe_list.append(new_expe)

    return extended_expe_list


class LoggingPrinter:
    def __init__(self, expe_path):
        log_file_path = os.path.join(expe_path, "logfile.log")
        self.out_file = open(log_file_path, "w")
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        #this object will take over `stdout`'s job
        sys.stdout = self
        sys.stderr = self
    #executed when the user does a `print`
    def write(self, text):
        self.old_stdout.write(text)
        self.out_file.write(text)
    def flush(self):
        pass
    #executed when `with` block begins
    def __enter__(self):
        return self
    #executed when `with` block ends
    def __exit__(self, error_type, value, traceback):
        #we don't want to log anymore. Restore the original stdout object.
        sys.stdout = self.old_stdout
        if traceback is None:
            sys.stderr = self.old_stderr



# =====================
# OTHER RANDOM FUNCTION
# =====================
def set_seed(seed):
    import torch
    import random
    import numpy as np

    if seed >= 0:
        print('Using seed {}'.format(seed))
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    else:
        raise NotImplementedError("Cannot set negative seed")


if __name__ == "__main__":

    with LoggingPrinter(''):
        for i in range(10):
            print("Test")

    print("C'est FINI, autre chose")

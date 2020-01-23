from tensorboardX import SummaryWriter
import numpy as np
import json
from os import path

class SweetLogger(SummaryWriter):
    def __init__(self, dump_step, path_to_log=None):
        """
        SweetLogger is a sweet tensorboardX logger
        At allows to store lists of variable and dump them easily without having to manually average and reset list

        Example of usage:

        my_sweet = SweetLogger('out_test/my_expe')

        my_sweet.log('reward', 10)
        my_sweet.log('reward', 20)
        my_sweet.dump(n_step=5)

        my_sweet.log('reward', 0)
        my_sweet.log('reward', 1)
        my_sweet.dump(n_step=10)

        out_test/my_expe contains :
            data/reward_mean : 15 at step 5 and 0.5 at step 10

        for each variable, you can specify which operation will be applied
        by sending a list of str the first time you call .log with this var

        my_sweet.log('reward', 0, operation=['max', 'mean'])
        my_sweet.log('reward', 10)
        my_sweet.dump(n_step=10)

        file contains : data/reward_max = 10,  data/reward_mean = 5
        """

        self.path_to_log = path_to_log
        super().__init__(path_to_log)

        # Each variable is a key
        # Each contains a list of values, and operation(s?) you want to apply
        self.variable_to_log = dict()
        self.sentence_to_log = dict()
        self.dump_step = dump_step
        self.str2op = {'mean': np.mean, 'max': np.max, 'min': np.min}

        self.next_dump_step = dump_step

    def log(self, key, value, operation='mean'):
        if key in self.variable_to_log:
            self.variable_to_log[key]['values'].append(value)
        else:
            self.variable_to_log[key] = dict()
            self.variable_to_log[key]['values'] = [value]
            self.variable_to_log[key]['operation'] = operation if type(operation) is list else [operation]

    def dump(self, total_step, train=True):
        """
        Dump all tensorboard data in one pass, empty temporary storage in the end
        :param total_step:
        :return:
        """
        if total_step > self.next_dump_step:
            # Dump sentences
            file_name = "sentences_count_step_{}_{}.json".format(self.next_dump_step, 'train' if train else 'test')
            sentences_path = path.join(self.path_to_log, file_name)
            json.dump(self.sentence_to_log, fp=open(sentences_path, 'w'), indent='    ', separators=('',':'))

            # Dump variables
            for variable_name, var_dict in self.variable_to_log.items():
                for op in var_dict['operation']:
                    operation_to_apply = self.str2op[op]
                    if len(var_dict['values']) == 0:
                        break
                    value = operation_to_apply(var_dict['values'])
                    self.add_scalar(variable_name + '_' + op, value, self.next_dump_step)

            self.reset()
            self.next_dump_step += self.dump_step
            return True
        return False


    def reset(self):
        # Reset instructions
        self.sentence_to_log = dict()

        # Reset variables
        for key in self.variable_to_log.keys():
            self.variable_to_log[key]['values'] = []

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        super().add_image(tag, img_tensor, global_step, walltime, dataformats)

    def store_sentences(self, sentence, reward):

        reward = str(reward)
        if sentence not in self.sentence_to_log:
            self.sentence_to_log[sentence] = dict()

        if reward not in self.sentence_to_log[sentence]:
            self.sentence_to_log[sentence][reward] = 0

        self.sentence_to_log[sentence][reward] += 1
import copy
import matplotlib.pyplot as plt
import seaborn
seaborn.set_palette('colorblind')

import random

import numpy as np

class QValueVisualizer(object):

    def __init__(self, ep_num_to_log=None, proba_log=None):

        assert ep_num_to_log or proba_log, "Need at least ep_num_to_log or proba_log"

        self.ep_num_to_log = ep_num_to_log
        self.proba_log = proba_log


    def render_state_and_q_values(self, game, q_values, ep_num):

        if self.ep_num_to_log:
            storing_this_step = ep_num in self.ep_num_to_log
        else:
            storing_this_step = random.random() < self.proba_log

        if storing_this_step:

            n_action = game.action_space.n
            max_action = np.max(q_values)

            fig = plt.figure()
            fig.add_subplot(121)

            plt.imshow(game.render('rgb_array'))

            f = plt.gcf()
            f.set_size_inches(9, 5)

            fig.add_subplot(122)

            plt.bar(x=list(range(n_action)),
                    height=q_values,
                    color=[(0.1, 0.2, 0.8) if i != max_action else (0.8, 0.1, 0.1) for i in
                           range(game.action_space.n)],
                    tick_label=[str(l) for l in range(n_action)])

            plt.xticks(fontsize=10, rotation=70)

            plt.xlabel('action', fontsize=16)
            plt.ylabel('q_value', fontsize=16)

            plt.tight_layout()

            fig.canvas.draw()
            array_rendered = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            array_rendered = array_rendered.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close()
            # X = np.array(fig.canvas)

            return array_rendered

        else:
            return
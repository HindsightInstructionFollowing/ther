from algo.replay_buffer import AbstractReplay
from algo.neural_architecture import InstructionGenerator


class LearntHindsightExperienceReplay(AbstractReplay):
    def __init__(self, input_shape, n_output, config, device):
        super().__init__(config)

        self.instruction_generator = InstructionGenerator(input_shape=input_shape,
                                                          n_output=n_output,
                                                          config=config["generator_model"],
                                                          device=device)

        self.update_frequency =                config["update_frequency"]
        self.n_sample_before_using_generator = config["n_sample_before_using_generator"]
        self.count_sample = 1

    def add_transition(self, curr_state, action, reward, next_state, terminal, mission, mission_length, hindsight_mission=None):
        self.current_episode.append(
            self.transition(curr_state, action, reward, next_state, terminal, mission, mission_length)
        )
        if self.count_sample % self.update_frequency == 0:
            self._train_generator()

        if terminal and reward <= 0 and self.count_sample > self.n_sample_before_using_generator:
            assert hindsight_mission is not None, \
                "Environment didn't provide hindsight mission, weird ! (Even though it's not used here)"

            hindsight_mission = None
            last_transition = self.current_episode[-1]
            last_state = last_transition.current_state

            # todo : in the long run, should take the whole trajectory as input
            hindsight_mission = self.instruction_generator.generate(last_state) # check last state represent what you want

            # Substitute the old mission with the new one, change the reward too
            hindsight_episode = [self.transition(st, a, self.hindsight_reward, st_plus1, end_ep, hindsight_mission,
                                                 len(hindsight_mission))
                                 for st, a, wrong_reward, st_plus1, end_ep, wrong_mission, length in
                                 self.current_episode
                                 ]

            self._store_episode(hindsight_episode)
            self._store_episode(self.current_episode)
            self.current_episode = []

    def _train_generator(self):
        pass

    def sample(self, batch_size):
        pass

    def update_transitions_proba(self):
        pass
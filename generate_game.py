import gym

env = gym.make("MiniGrid-Fetch-5x5-N2-v0")

save_dict = dict()
n_episodes = 10000

for i in range(n_episodes):
    obs = env.reset()
    grid = env.grid
    mission = env.mission
    done = False

    while not done:
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)


    # Done, goes here

    # Store objective
    if reward > 0:
        pass
from Snaky import *
from Agent import *


# constants
BLOCK_SIDE = 30
HEIGHT = WIDTH = 20

MAX_STEPS_PER_EPISODE = 10000

print('TEST')

NUM_TEST_EPISODES = 50
DISPLAY_TEST = True

game = gameENV(HEIGHT, WIDTH, BLOCK_SIDE)  # initializing game
tester = testAgn()

game.initialize_display()

tot_score = np.zeros(shape=(NUM_TEST_EPISODES,))  # total score
n_timesteps = np.zeros(shape=(NUM_TEST_EPISODES,))

for i_episode_test in range(NUM_TEST_EPISODES):
    # Initialize the environment and get it's state
    state = game.reset()
    for timestep in range(1, MAX_STEPS_PER_EPISODE):
        action = tester.select_action(state, game.snakeDirection)
        next_state, done = game.test_step(action)

        # Move to the next state
        state = next_state

        if DISPLAY_TEST:
            game.update_game_screen()

        if done:
            n_timesteps[i_episode_test] = timestep
            tot_score[i_episode_test] = game.score
            break

    print('episode', i_episode_test)

print(game.REWARDS.values())
print('number of test episodes: ' , NUM_TEST_EPISODES)
print('max score:', np.max(tot_score))
print('average score: ', (np.sum(tot_score)/NUM_TEST_EPISODES))
print('average timesteps: ', (np.sum(n_timesteps)/NUM_TEST_EPISODES))
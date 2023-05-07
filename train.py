from collections import namedtuple
import numpy as np
from Snaky import *
from Agent import *


# defining transition
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# constants
BLOCK_SIDE = 30
HEIGHT = WIDTH = 20

CLOSERREWARD = 1
FURTHERREWARD = -1
APPLEREWARD = 20
DEATHREWARD = -500
WONREWARD = 300

DISPLAY = False  # whether to display gui
NUM_EPISODES = 20  # number of rounds
MAX_STEPS_PER_EPISODE = 1000  # if snake doesn't die before max steps, the environment resets anyway
BATCH_SIZE = 800  #batch size for training
MAX_REPLAY_MEMORY_LEN = 1500
EPSILON = 1  # beginning value of epsilon
EPSILON_END = 0.01  # MIN value of epsilon
EPSILON_DECAY = 0.95
GAMMA = 0.99

NEURONS = [100, 100, 100]  # neurons for every dense layer
OPTIMIZER_LEARNING_RATE = 1e-4

UPDATE_TARGET_NET = 20  # update target net after 10 episodes

title = 'linear run'


game = gameENV(HEIGHT, WIDTH, BLOCK_SIDE)  # initializing game
game.set_rewards(APPLEREWARD, DEATHREWARD, WONREWARD, CLOSERREWARD, FURTHERREWARD)  # defining rewards

RLAgent = agent(BATCH_SIZE, MAX_REPLAY_MEMORY_LEN, OPTIMIZER_LEARNING_RATE, NEURONS)

if DISPLAY:
    game.initialize_display()

# saving the history of the agent (for later plotting) : total rewards per episode, total steps per episode, total score per episode(how many apples the agent ate)
tot_rewards = np.zeros(shape=(NUM_EPISODES,))  # total rewards for each episode
tot_score = np.zeros(shape=(NUM_EPISODES,))  # total score
n_timesteps = np.zeros(shape=(NUM_EPISODES,))  # number of timesteps until done for each episode

for i_episode in range(NUM_EPISODES):
    print('episode', i_episode)

    # Initialize the environment and get it's state
    state = game.reset()

    for timestep in range(1, MAX_STEPS_PER_EPISODE):
        action, relative_action = RLAgent.select_action(EPSILON, state, game.snakeDirection)
        # print('action', action)
        next_state, reward, done = game.step(action)
        # print('next_S', next_state, 'reward', reward, 'done', done)

        # decay epsilon
        EPSILON = RLAgent.decay_epsilon(EPSILON, EPSILON_END, EPSILON_DECAY)

        # Store the transition in memory
        transition = Transition(state, relative_action, next_state, reward, done)
        RLAgent.add_to_replayMemory(transition)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        RLAgent.train_model(GAMMA)

        # add reward to the rewards of episode
        tot_rewards[i_episode] += reward


        if DISPLAY:
            game.update_game_screen()

        if done:
            # save number of timesteps for this episode
            n_timesteps[i_episode] = timestep

            # save score for this episode
            print('score', game.score)
            tot_score[i_episode] = game.score

            # exit while
            break

    # update the target network weights
    if (i_episode % UPDATE_TARGET_NET == 0):
        RLAgent.update_target_net()

        # RLAgent.save_model()


RLAgent.save_model()
plot_training(n_timesteps, tot_rewards, tot_score, GAMMA, EPSILON_DECAY, game.REWARDS, RLAgent.BATCH_SIZE, len(RLAgent.replay_memory), UPDATE_TARGET_NET, title)
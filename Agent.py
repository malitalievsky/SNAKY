import matplotlib.pyplot as plt
import numpy as np
import random
import PIL

from collections import deque

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam

from Snaky import *

DIRECTIONS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])

class agent():
    def __init__(self, BATCH_SIZE, MAX_MEMORY_LEN, optimizerLR, neurons):
        self.P_model = self.create_linearNet(neurons, optimizerLR)  # policy
        self.target_model = self.create_linearNet(neurons, optimizerLR)  # target
        self.update_target_net()  # updates weights
        self.replay_memory = deque([], maxlen= MAX_MEMORY_LEN)
        self.BATCH_SIZE = BATCH_SIZE


    def create_linearNet(self, neurons_for_each_layer, optimizerLR, n_observations = 11, n_possible_actions = 3): #the snake can only move right, left and straight from its point, therefore there are only 3 actions available
        model = Sequential([
            layers.Input(shape=(n_observations,)),
            layers.Dense(neurons_for_each_layer[0], activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(neurons_for_each_layer[1], activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(neurons_for_each_layer[2], activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_possible_actions, activation='softmax')
        ])

        model.compile(loss='mse', optimizer=Adam(learning_rate=optimizerLR, clipnorm=1.0))
        print(model.summary())
        return model

    def train_model(self, GAMMA):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < self.BATCH_SIZE:
            return

        # Get a minibatch of random samples from memory replay
        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        current_states = []
        actions =[]
        next_states = []
        rewards = []
        done_history =[]
        for transition in batch:  # transition is a named tuple
            current_states.append(transition.state)
            actions.append(transition.action)
            next_states.append(transition.next_state)
            rewards.append(transition.reward)
            done_history.append(transition.done)

        # convert memory replay sample into np arrays
        current_states = np.array(current_states)
        actions = np.array(actions).transpose()
        next_states = np.array(next_states)
        rewards = np.array(rewards)
        done_history = np.array(done_history)

        # predict target q values
        target_q_values = rewards + GAMMA * (np.amax(self.target_model.predict_on_batch(next_states), axis=1)) * (1 - (done_history * 1))

        # create a targets array of the same shape as the policy q values array
        targets_full = self.P_model.predict_on_batch(current_states)

        indexes = np.arange(self.BATCH_SIZE).transpose()
        targets_full[[indexes], [actions]] = target_q_values

        self.P_model.fit(current_states, targets_full, epochs=1, verbose=0)


    def update_target_net(self):
        # update the target network with new weights
        self.target_model.set_weights(self.P_model.get_weights())


    def add_to_replayMemory(self, transition):
        self.replay_memory.append(transition)


    # decaying epsilon as the agent becomes more experienced
    def decay_epsilon(self, epsilon, MIN_EPSILON, EPSILON_DECAY): # epsilon max: starting value of epsilon, epsilon min: final value of epsilon, epsilon decay: number beween 0-1, decaying epsilon(new epsilon = 0.9 of the old epsilon)
        if epsilon > MIN_EPSILON:
            epsilon *= EPSILON_DECAY
        new_eps = max(epsilon, MIN_EPSILON)
        return new_eps

    # the agent selects and returns an action based on epsilon
    def select_action(self, epsilon, state, curr_direction):
        randProb = random.random()  # random.random generates number between 0-1
        best_action_relative = np.array(0)
        if randProb < epsilon:
            allowed_dir_from_currPos = np.delete(DIRECTIONS, np.where(((-1) * curr_direction)), 0)

            action = random.choice(allowed_dir_from_currPos)
        else:
            # action_q_values = self.P_model(state, training= False)
            relative_action_q_values = self.P_model.predict(np.array([state]), verbose=0)
            # find best action
            best_action_relative = tf.argmax(relative_action_q_values[0]).numpy()

            if best_action_relative == 0:
                action = curr_direction
            if best_action_relative == 1:
                action = curr_direction @  np.array([[0, 1], [-1, 0]])
            if best_action_relative == 2:
                action = curr_direction @ np.array([[0, -1], [1, 0]])

        return action, best_action_relative  # the action returned is [(1, 0), (-1, 0), (0, 1), (0, -1)] and abolute (does not depend on the snake current dir)


    def save_model(self):
        self.P_model.save('policy.h5')
        self.target_model.save('target.h5')


    def load_model(self):
        self.P_model = tf.keras.models.load_model('policy.h5')
        self.target_model = tf.keras.models.load_model('target.h5')



#### TESTER #################

class testAgn():
    def __init__(self):
        # load trained models
        self.P_model = tf.keras.models.load_model('policy.h5')
        self.target_model = tf.keras.models.load_model('target.h5')

    def select_action(self, state, curr_dir):
        relative_action_qs = self.P_model.predict(np.array([state]), verbose=0)
        # find best action
        best_relative_action = np.argmax(relative_action_qs[0])
        if best_relative_action == 0:
            action = curr_dir
        if best_relative_action == 1:
            action = curr_dir @ np.array([[0, 1], [-1, 0]])  # rotate right
        if best_relative_action == 2:
            action = curr_dir @ np.array([[0, -1], [1, 0]])  # rotate left
        return action



##############################

# plotting
def moving_average(data, window_size = 50):
    moving_averages = []
    for i in range(data.shape[0] - window_size + 1) :
        window = data[i : i + window_size]
        window_average = np.sum(window) / window_size
        moving_averages.append(window_average)

    return moving_averages


def plot_training(timesteps, rewards, score, gamma, epsilon_decay, reward_dict, batch_size, replay_memory_len, updateAfter, title):
    fig, (tot_timesteps, tot_rewards, tot_score) = plt.subplots(1, 3, figsize=(15, 7))

    tot_timesteps.plot(timesteps, color= 'C0')
    tot_timesteps.set_title('Timesteps till done')
    tot_timesteps.set_xlabel('Episode')
    tot_timesteps.set_ylabel('Timesteps')

    tot_rewards.plot(rewards, color= 'C0')
    print(rewards.shape[0])
    tot_rewards.plot(moving_average(rewards), color= 'red')
    tot_rewards.set_title('Total Rewards per game')
    tot_rewards.set_xlabel('Episode')
    tot_rewards.set_ylabel('Reward')

    tot_score.plot(score, color= 'C0')
    tot_score.plot(moving_average(score), color= 'red')
    tot_score.set_title('Score obtained each game')
    tot_score.set_xlabel('Episode')
    tot_score.set_ylabel('Score')

    batchsize_t = 'Batch size = ' + str(batch_size)
    ReplayMS_t = ', Replay Memory size = ' + str(replay_memory_len)
    num_Eps_t = ', NUM Episodes = ' + str(len(timesteps))
    gamma_t = ', GAMMA = ' + str(gamma)
    epsDecay_t = ', Epsilon Decay rate = ' + str(epsilon_decay)
    rewards_t = ', Rewards: ' + str(reward_dict)
    update_t = ', Update target net after ' + str(updateAfter)
    plt.figtext(0.01, 0.01, (batchsize_t + ReplayMS_t + num_Eps_t + gamma_t + epsDecay_t + rewards_t + update_t))

    fig.suptitle(title)
    plt.show()
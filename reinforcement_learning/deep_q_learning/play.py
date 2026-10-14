#!/usr/bin/env python3
"""
Evaluation script to render and play Atari Breakout using a trained DQN agent.
"""

import sys
import tensorflow as tf
import keras
import tensorflow.keras

if not hasattr(tensorflow.keras, '__version__'):
    tensorflow.keras.__version__ = keras.__version__
sys.modules['tensorflow.keras'] = tensorflow.keras
sys.modules['tensorflow.keras'].__version__ = keras.__version__

import gymnasium as gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from rl.memory import SequentialMemory


def play_breakout():
    """
    Loads the saved policy network and evaluates the agent on Breakout
    with visual rendering.
    """
    env_name = "Breakout-v4"
    env = gym.make(env_name, render_mode="human")

    nb_actions = env.action_space.n

    # Reconstruct the exact model architecture
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    memory = SequentialMemory(limit=1000, window_length=4)
    policy = GreedyQPolicy()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        target_model_update=1000,
        policy=policy
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Load trained weights
    dqn.load_weights('policy.h5')

    # Test/Play the game
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    play_breakout()

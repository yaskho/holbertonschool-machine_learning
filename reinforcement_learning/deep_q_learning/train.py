#!/usr/bin/env python3
"""
Training script for an Atari Breakout agent using Deep Q-Learning.
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
import numpy as np
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor


class GymWrapper(gym.Wrapper):
    """
    Wrapper to make Gymnasium environment compatible with keras-rl2.
    """

    def reset(self, **kwargs):
        """Resets the environment and returns only the observation."""
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        """Executes an action and returns standard 4 elements."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info


class AtariProcessor(Processor):
    """
    A processor to resize and preprocess Atari frames for keras-rl2 compatibility.
    """

    def process_observation(self, observation):
        """Processes an observation before it is given to the agent."""
        if isinstance(observation, tuple):
            observation = observation[0]
        img = Image.fromarray(observation)
        img = img.resize((84, 84)).convert('L')
        processed_observation = np.array(img)
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        """Scales a batch of states before passing it to the network."""
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        """Clips rewards between -1 and 1 for stability."""
        return np.clip(reward, -1., 1.)


def train_breakout():
    """
    Initializes the Breakout environment, constructs the DQN model,
    trains the agent, and saves the final policy weights.
    """
    env_name = "Breakout-v4"
    env = gym.make(env_name)
    env = GymWrapper(env)
    np.random.seed(0)
    env.reset(seed=0)

    nb_actions = env.action_space.n

    # Build the CNN model for Atari game frames (shape: 4 x 84 x 84)
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))

    memory = SequentialMemory(limit=1000000, window_length=4)
    policy = EpsGreedyQPolicy(eps=0.1)
    processor = AtariProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=memory,
        nb_steps_warmup=50000,
        target_model_update=10000,
        policy=policy,
        processor=processor
    )
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Train the model
    dqn.fit(env, nb_steps=1750000, visualize=False, verbose=2)

    # Save the trained policy weights
    dqn.save_weights('policy.h5', overwrite=True)


if __name__ == '__main__':
    train_breakout()

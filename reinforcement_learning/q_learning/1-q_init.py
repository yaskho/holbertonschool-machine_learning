#!/usr/bin/env python3
"""
Contains function to initialize a Q-table for a Gymnasium environment.
"""
import numpy as np


def q_init(env):
    """
    Initializes the Q-table with zeros based on the environment's state
    and action space sizes.

    Args:
        env: The FrozenLakeEnv instance

    Returns:
        The Q-table as a numpy.ndarray of zeros with shape (states, actions)
    """
    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros((state_space_size, action_space_size))

    return q_table

#!/usr/bin/env python3
"""
Contains the epsilon_greedy function for action selection.
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using the epsilon-greedy strategy.

    Args:
        Q: numpy.ndarray containing the q-table
        state: current state index
        epsilon: epsilon value for exploration vs exploitation

    Returns:
        The index of the selected action
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        action = np.random.randint(0, Q.shape[1])
    else:
        action = np.argmax(Q[state])

    return action

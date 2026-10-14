#!/usr/bin/env python3
"""
Contains function to play an episode using a trained Q-table
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Has the trained agent play an episode in FrozenLake environment.

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: total rewards for the episode
        rendered_outputs: list of rendered board states at each step
    """
    state, _ = env.reset()
    rendered_outputs = [env.render()]
    total_rewards = 0.0

    for step in range(max_steps):
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        total_rewards += reward
        rendered_outputs.append(env.render())

        if terminated or truncated:
            break

    return total_rewards, rendered_outputs

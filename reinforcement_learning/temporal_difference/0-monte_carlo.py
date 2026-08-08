#!/usr/bin/env python3
"""
Contains function to perform Monte Carlo policy evaluation.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for value estimation.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes a state and returns next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        episode = []
        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state
            if terminated or truncated:
                break

        G = 0
        states = [step[0] for step in episode]
        for i in range(len(episode) - 1, -1, -1):
            state, reward = episode[i]
            G = gamma * G + reward
            if state not in states[:i]:
                V[state] += alpha * (G - V[state])

    return V
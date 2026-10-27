#!/usr/bin/env python3
"""Module that contains the Monte Carlo algorithm for policy evaluation."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    Args:
        env: gymnasium environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns the next action
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

            if terminated or truncated:
                break

            state = next_state

        G = 0
        visited = set()
        for t in range(len(episode) - 1, -1, -1):
            s, r = episode[t]
            G = gamma * G + r

            # Check if this state was visited earlier in this episode
            first_visit = True
            for k in range(t):
                if episode[k][0] == s:
                    first_visit = False
                    break

            if first_visit:
                V[s] = V[s] + alpha * (G - V[s])

    return V

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
    n_states = V.shape[0]

    for episode in range(episodes):
        state, _ = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        episode_data = np.array(episode_data, dtype=object)
        G = 0
        visited_states = set()

        for state, reward in reversed(episode_data):
            G = gamma * G + reward

            # First-visit Monte Carlo update
            if state not in visited_states:
                visited_states.add(state)
                V[state] = V[state] + alpha * (G - V[state])

    return V

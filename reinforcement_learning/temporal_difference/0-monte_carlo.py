#!/usr/bin/env python3
"""Module containing the Monte Carlo policy evaluation algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    Args:
        env: gymnasium environment instance
        V: numpy.ndarray of shape (s,) containing the value estimate
        policy: function that takes in a state and returns next action
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        state, _ = env.reset()
        states = []
        rewards = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(state)
            rewards.append(reward)

            if terminated or truncated:
                break

            state = next_state

        G = 0
        for t in range(len(states) - 1, -1, -1):
            G = gamma * G + rewards[t]
            if states[t] not in states[:t]:
                V[states[t]] = V[states[t]] + alpha * (G - V[states[t]])

    return V

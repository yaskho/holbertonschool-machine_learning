Here is the complete implementation for Task 0: Monte Carlo prediction algorithm, adhering to all style, documentation, and technical requirements for Python 3.9, NumPy 1.25.2, and Gymnasium 0.29.1.

File: 0-monte_carlo.py
Python
#!/usr/bin/env python3
"""
Monte Carlo Algorithm module for Reinforcement Learning.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Performs the Monte Carlo algorithm for policy evaluation.

    Parameters:
    - env: environment instance
    - V: numpy.ndarray of shape (s,) containing the value estimate
    - policy: a function that takes in a state and returns the next action
    - episodes: total number of episodes to train over (default 5000)
    - max_steps: maximum number of steps per episode (default 100)
    - alpha: learning rate (default 0.1)
    - gamma: discount rate (default 0.99)

    Returns:
    - V: the updated value estimate numpy.ndarray
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

        states_in_ep = [x[0] for x in episode]
        rewards_in_ep = [x[1] for x in episode]

        T = len(episode)
        returns = [0] * T
        g = 0
        for t in range(T - 1, -1, -1):
            g = gamma * g + rewards_in_ep[t]
            returns[t] = g

        seen = set()
        for t in range(T):
            s = states_in_ep[t]
            if s not in seen:
                seen.add(s)
                V[s] += alpha * (returns[t] - V[s])

    return V

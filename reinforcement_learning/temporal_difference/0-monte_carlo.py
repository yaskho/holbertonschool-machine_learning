#!/usr/bin/env python3
"""
Monte Carlo Algorithm module for Reinforcement Learning.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
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
        reset_res = env.reset()
        if isinstance(reset_res, tuple):
            state = reset_res[0]
        else:
            state = reset_res

        episode = []

        for _ in range(max_steps):
            action = policy(state)
            step_res = env.step(action)
            if len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
            else:
                next_state, reward, done, _ = step_res
                terminated = done
                truncated = False

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

        for t in range(T):
            s = states_in_ep[t]
            V[s] += alpha * (returns[t] - V[s])

    return V

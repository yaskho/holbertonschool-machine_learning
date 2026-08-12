#!/usr/bin/env python3
"""
Monte Carlo Policy Evaluation module.
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to update the value estimate.
    
    Parameters:
    - env: the openAI gymnasium environment instance
    - V: numpy.ndarray of shape (s,) containing the value estimate
    - policy: a function that takes in a state and returns an action
    - episodes: total number of episodes to train over
    - max_steps: maximum number of steps per episode
    - alpha: the learning rate
    - gamma: the discount rate
    
    Returns:
    - V: the updated value estimate
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
        states = [x[0] for x in episode]
        for t in range(len(episode) - 1, -1, -1):
            state, reward = episode[t]
            G = gamma * G + reward
            # First-visit Monte Carlo update
            if state not in states[:t]:
                V[state] = V[state] + alpha * (G - V[state])

    return V

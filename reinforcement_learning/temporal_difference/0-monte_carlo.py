#!/usr/bin/env python3
"""Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs the Monte Carlo algorithm.

    Args:
        env: Environment instance.
        V: Value estimate.
        policy: Policy function.
        episodes: Number of episodes.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        Updated value estimate V.
    """
    for _ in range(episodes):
        state = env.reset()[0]
        episode = []

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, reward))
            state = next_state

            if terminated or truncated:
                break

        G = 0
        visited = set()

        for state, reward in reversed(episode):
            G = reward + gamma * G

            if state not in visited:
                V[state] += alpha * (G - V[state])
                visited.add(state)

    return V

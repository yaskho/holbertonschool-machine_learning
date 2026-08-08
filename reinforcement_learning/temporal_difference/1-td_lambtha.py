#!/usr/bin/env python3
"""TD(lambda) algorithm."""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """Performs the TD(lambda) algorithm.

    Args:
        env: Environment instance.
        V: Value estimate.
        policy: Policy function.
        lambtha: Eligibility trace factor.
        episodes: Number of episodes.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        Updated value estimate V.
    """
    for _ in range(episodes):
        state = env.reset()[0]
        eligibility = np.zeros(V.shape)

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                delta = reward - V[state]
            else:
                delta = reward + gamma * V[next_state] - V[state]

            eligibility[state] += 1

            V += alpha * delta * eligibility
            eligibility *= gamma * lambtha

            state = next_state

            if terminated or truncated:
                break

    return V

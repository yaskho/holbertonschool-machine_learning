#!/usr/bin/env python3
"""Monte Carlo algorithm."""


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
        states = []
        rewards = []

        for _ in range(max_steps):
            states.append(state)
            action = policy(state)
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        G = 0
        for i in range(len(states) - 1, -1, -1):
            G = rewards[i] + gamma * G
            state = states[i]
            V[state] += alpha * (G - V[state])

    return V

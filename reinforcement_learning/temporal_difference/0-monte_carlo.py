```python
#!/usr/bin/env python3
"""Monte Carlo algorithm."""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Performs Monte Carlo prediction on an environment.

    Args:
        env: The environment instance.
        V: A numpy.ndarray containing the value estimates.
        policy: Function that takes a state and returns an action.
        episodes: Number of episodes to train over.
        max_steps: Maximum number of steps per episode.
        alpha: Learning rate.
        gamma: Discount rate.

    Returns:
        The updated value estimate V.
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
        for state, reward in reversed(episode):
            G = gamma * G + reward
            V[state] = V[state] + alpha * (G - V[state])

    return V

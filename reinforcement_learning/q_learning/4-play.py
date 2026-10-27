#!/usr/bin/env python3
"""Module to play an episode using a trained Q-table."""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table (pure exploitation).

    Args:
        env: FrozenLakeEnv instance
        Q: numpy.ndarray containing the Q-table
        max_steps: maximum number of steps in the episode

    Returns:
        total_rewards: total reward obtained during the episode
        rendered_outputs: list of rendered strings representing board states
    """
    res = env.reset()
    state = res[0] if isinstance(res, tuple) else res

    rendered_outputs = []
    # Capture initial state rendering
    rendered_outputs.append(env.render())

    total_rewards = 0.0

    for step in range(max_steps):
        # Always exploit the Q-table
        action = np.argmax(Q[state])

        step_res = env.step(action)
        if len(step_res) == 5:
            next_state, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            next_state, reward, done, _ = step_res

        total_rewards += reward
        # Capture rendering after step
        rendered_outputs.append(env.render())

        if done:
            break

        state = next_state

    return total_rewards, rendered_outputs

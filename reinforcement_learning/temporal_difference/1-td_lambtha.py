#!/usr/bin/env python3
"""Module containing the TD(lambda) algorithm for policy evaluation."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(lambda) algorithm for policy evaluation.

    Args:
        env: environment instance
        V: numpy.ndarray of shape (s,) containing value estimates
        policy: function that takes in a state and returns next action
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate

    Returns:
        V: updated value estimate
    """
    for _ in range(episodes):
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        E = np.zeros_like(V)

        for _ in range(max_steps):
            action = policy(state)
            step_res = env.step(action)

            if len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_res

            # Update eligibility trace for current state
            E[state] += 1

            # TD Error computation
            delta = reward + gamma * V[next_state] - V[state]

            # Update all state value estimates based on eligibility traces
            V += alpha * delta * E

            # Decay eligibility traces
            E *= gamma * lambtha

            if done:
                break

            state = next_state

    return V

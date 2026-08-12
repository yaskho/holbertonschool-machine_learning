#!/usr/bin/env python3
"""
SARSA(lambda) algorithm implementation
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs SARSA(lambda) on a gymnasium environment

    Parameters:
        env: environment instance
        Q: numpy.ndarray of shape (s, a) containing the Q table
        lambtha: eligibility trace factor
        episodes: total number of episodes to train over
        max_steps: maximum number of steps per episode
        alpha: learning rate
        gamma: discount rate
        epsilon: initial threshold for epsilon greedy
        min_epsilon: minimum value that epsilon should decay to
        epsilon_decay: decay rate for updating epsilon between episodes

    Returns:
        Q: updated Q table
    """
    init_epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        E = np.zeros_like(Q)

        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, Q.shape[1])
        else:
            action = np.argmax(Q[state])

        for _ in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if np.random.uniform(0, 1) < epsilon:
                next_action = np.random.randint(0, Q.shape[1])
            else:
                next_action = np.argmax(Q[next_state])

            if done:
                delta = reward - Q[state, action]
            else:
                delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            E[state, action] += 1
            Q += alpha * delta * E
            E *= gamma * lambtha

            if done:
                break

            state = next_state
            action = next_action

        epsilon = min_epsilon + (init_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return Q

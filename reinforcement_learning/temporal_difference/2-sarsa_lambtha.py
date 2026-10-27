#!/usr/bin/env python3
"""Module containing the SARSA(lambda) algorithm."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    Performs the SARSA(lambda) algorithm to update a Q-table.

    Args:
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
    initial_epsilon = epsilon
    n_states, n_actions = Q.shape

    def epsilon_greedy(state, eps):
        """Selects an action using the epsilon-greedy strategy."""
        p = np.random.uniform()
        if p < eps:
            return np.random.randint(n_actions)
        return np.argmax(Q[state])

    for episode in range(episodes):
        res = env.reset()
        state = res[0] if isinstance(res, tuple) else res
        E = np.zeros((n_states, n_actions))

        action = epsilon_greedy(state, epsilon)

        for _ in range(max_steps):
            step_res = env.step(action)

            if len(step_res) == 5:
                next_state, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_res

            next_action = epsilon_greedy(next_state, epsilon)

            # TD Error computation for SARSA
            delta = reward + gamma * Q[next_state, next_action] - Q[state, action]

            # Update eligibility trace
            E[state, action] += 1

            # Update Q-table with eligibility traces
            Q += alpha * delta * E

            # Decay eligibility traces
            E *= gamma * lambtha

            if done:
                break

            state = next_state
            action = next_action

        # Decay epsilon after each episode
        epsilon = min_epsilon + (initial_epsilon - min_epsilon) * np.exp(
            -epsilon_decay * episode
        )

    return Q

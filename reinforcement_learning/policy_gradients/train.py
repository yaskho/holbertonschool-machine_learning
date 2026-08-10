#!/usr/bin/env python3
"""
Module to train a Policy Gradient (REINFORCE) agent with optional rendering.
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements full training using Monte-Carlo policy gradient (REINFORCE).

    Args:
        env: Initial environment.
        nb_episodes (int): Number of episodes used for training.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        show_result (bool): Render environment every 1000 episodes if True.

    Returns:
        list: All values of score (sum of all rewards per episode).
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        grads = []
        rewards = []

        while True:
            if show_result and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            grads.append(grad)
            rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break

        score = sum(rewards)
        scores.append(score)

        G = 0
        for i in reversed(range(len(rewards))):
            G = rewards[i] + gamma * G
            weight += alpha * grads[i] * G

        print("Episode: {} Score: {}".format(episode, score))

    return scores

#!/usr/bin/env python3
"""
Module to compute policy probabilities and Monte-Carlo policy gradients.
"""
import numpy as np


def policy(matrix, weight):
    """
    Computes the policy with a weight of a matrix.

    Args:
        matrix (np.ndarray): The state array or matrix.
        weight (np.ndarray): The weight matrix.

    Returns:
        np.ndarray: Softmax probabilities for each action.
    """
    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]
    z = np.dot(matrix, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state
    and weight matrix.

    Args:
        state (np.ndarray): Current observation of the environment.
        weight (np.ndarray): Matrix of random weights.

    Returns:
        tuple: (action, gradient) where action is sampled action index
               and gradient is computed policy gradient matrix.
    """
    if state.ndim == 1:
        state = state[np.newaxis, :]

    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs[0])

    dsoftmax = probs.copy()
    dsoftmax[0, action] -= 1
    grad = np.dot(state.T, -dsoftmax)

    return action, grad

#!/usr/bin/env python3
"""
Module to compute policy probabilities using softmax.
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
    z = np.dot(matrix, weight)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

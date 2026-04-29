#!/usr/bin/env python3
"""Module for L2 regularization cost"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculates the cost with L2 regularization

    Args:
        cost (float): original cost
        lambtha (float): regularization parameter
        weights (dict): dictionary of weights and biases
        L (int): number of layers
        m (int): number of data points

    Returns:
        float: regularized cost
    """
    l2_sum = 0

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_sum += np.sum(W ** 2)

    return cost + (lambtha / (2 * m)) * l2_sum

#!/usr/bin/env python3
"""Module for gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights using gradient descent with L2 regularization

    Args:
        Y (numpy.ndarray): one-hot labels (classes, m)
        weights (dict): weights and biases
        cache (dict): activations
        alpha (float): learning rate
        lambtha (float): L2 parameter
        L (int): number of layers
    """
    m = Y.shape[1]
    weights_copy = weights.copy()

    # Last layer
    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]

        # L2 applied here
        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            A_prev = cache['A' + str(i - 1)]
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)

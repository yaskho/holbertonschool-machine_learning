#!/usr/bin/env python3
"""
Module to update weights and biases using gradient descent
with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient
    descent with L2 regularization.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) containing
           the correct labels for the data
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs of each layer of the neural network
        alpha: learning rate
        lambtha: L2 regularization parameter
        L: number of layers of the network
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)

        W = weights[W_key]

        dW = (1 / m) * np.matmul(dz, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            dz = np.matmul(W.T, dz) * (1 - (A_prev ** 2))

        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db

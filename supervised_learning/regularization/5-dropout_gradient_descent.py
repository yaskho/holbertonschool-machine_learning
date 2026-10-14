#!/usr/bin/env python3
"""
Module to update weights of a neural network with Dropout regularization
using gradient descent.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Args:
        Y: one-hot numpy.ndarray of shape (classes, m) containing
           the correct labels for the data
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs and dropout masks of each layer
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W_key = 'W' + str(i)
        b_key = 'b' + str(i)

        W = weights[W_key]

        dW = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        if i > 1:
            dz = np.matmul(W.T, dz)
            dz *= cache['D' + str(i - 1)]
            dz /= keep_prob
            dz *= (1 - (A_prev ** 2))

        weights[W_key] -= alpha * dW
        weights[b_key] -= alpha * db

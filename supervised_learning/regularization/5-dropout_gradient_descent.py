#!/usr/bin/env python3
"""Module for gradient descent with dropout"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Updates weights using gradient descent with dropout

    Args:
        Y (numpy.ndarray): one-hot labels (classes, m)
        weights (dict): weights and biases
        cache (dict): activations and dropout masks
        alpha (float): learning rate
        keep_prob (float): probability to keep neuron
        L (int): number of layers
    """
    m = Y.shape[1]
    weights_copy = weights.copy()

    # Last layer
    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            dZ = np.matmul(W.T, dZ)
            A_prev = cache['A' + str(i - 1)]

            # tanh derivative
            dZ = dZ * (1 - A_prev ** 2)

            # Apply dropout mask
            D = cache['D' + str(i - 1)]
            dZ = dZ * D

            # Scale
            dZ = dZ / keep_prob

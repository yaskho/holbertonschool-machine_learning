#!/usr/bin/env python3
"""Module for forward propagation with dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Performs forward propagation using dropout

    Args:
        X (numpy.ndarray): input data (nx, m)
        weights (dict): weights and biases
        L (int): number of layers
        keep_prob (float): probability of keeping a neuron

    Returns:
        dict: cache of activations and dropout masks
    """
    cache = {}
    cache['A0'] = X

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        # Last layer → softmax
        if i == L:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            cache['A' + str(i)] = A
        else:
            # tanh activation
            A = np.tanh(Z)

            # Dropout mask
            D = np.random.binomial(1, keep_prob, size=A.shape)

            # Apply dropout
            A = A * D

            # Scale
            A = A / keep_prob

            cache['A' + str(i)] = A
            cache['D' + str(i)] = D

    return cache

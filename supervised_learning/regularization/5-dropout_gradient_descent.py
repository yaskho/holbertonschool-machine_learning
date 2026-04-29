#!/usr/bin/env python3
"""Dropout Gradient Descent"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates weights with dropout"""
    m = Y.shape[1]
    weights_copy = weights.copy()

    # Output layer
    A_L = cache["A" + str(L)]
    dZ = A_L - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(l - 1)]
        W = weights_copy["W" + str(l)]

        # Gradients
        dW = (dZ @ A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update
        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db

        if l > 1:
            A_prev = cache["A" + str(l - 1)]
            dZ = W.T @ dZ

            # tanh derivative
            dZ = dZ * (1 - A_prev ** 2)

            # Apply dropout
            D = cache["D" + str(l - 1)]
            dZ = dZ * D
            dZ = dZ / keep_prob

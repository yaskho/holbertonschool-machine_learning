#!/usr/bin/env python3
"""Dropout Gradient Descent"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates weights using dropout gradient descent"""
    m = Y.shape[1]
    weights_copy = weights.copy()

    # Output layer
    A_L = cache["A" + str(L)]
    dZ = A_L - Y

    for l in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(l - 1)]
        W = weights_copy["W" + str(l)]

        # Compute gradients
        dW = (dZ @ A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights
        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db

        if l > 1:
            # Backprop using ORIGINAL weights
            dZ = W.T @ dZ

            A_prev = cache["A" + str(l - 1)]

            # tanh derivative
            dZ = dZ * (1 - A_prev ** 2)

            # Apply dropout
            D = cache["D" + str(l - 1)]
            dZ = dZ * D
            dZ = dZ / keep_prob

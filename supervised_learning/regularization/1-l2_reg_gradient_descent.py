#!/usr/bin/env python3
"""Module for gradient descent with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates weights using gradient descent with L2 regularization"""
    m = Y.shape[1]
    weights_copy = weights.copy()

    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights_copy['W' + str(i)]  # ← IMPORTANT

        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)

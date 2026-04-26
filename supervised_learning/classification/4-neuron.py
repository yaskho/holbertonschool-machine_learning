#!/usr/bin/env python3
"""Neuron class for binary classification"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """Initialize the neuron"""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weight vector"""
        return self.__W

    @property
    def b(self):
        """Bias"""
        return self.__b

    @property
    def A(self):
        """Activated output"""
        return self.__A

    def forward_prop(self, X):
        """Forward propagation"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Compute logistic regression cost"""
        m = Y.shape[1]
        return - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """
        Evaluate predictions

        Returns:
        prediction (1, m), cost
        """
        A = self.forward_prop(X)

        # Convert probabilities to binary predictions
        prediction = np.where(A >= 0.5, 1, 0)

        cost = self.cost(Y, A)

        return prediction, cost

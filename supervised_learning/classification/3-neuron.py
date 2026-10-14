#!/usr/bin/env python3
"""Neuron class for binary classification"""

import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification"""

    def __init__(self, nx):
        """
        Initialize the neuron

        Parameters:
        nx (int): number of input features
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # weights initialized with random normal distribution
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Weight vector getter"""
        return self.__W

    @property
    def b(self):
        """Bias getter"""
        return self.__b

    @property
    def A(self):
        """Activated output getter"""
        return self.__A

    def forward_prop(self, X):
        """
        Perform forward propagation

        Parameters:
        X (numpy.ndarray): input data of shape (nx, m)

        Returns:
        Activated output
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))  # sigmoid
        return self.__A

    def cost(self, Y, A):
        """
        Calculate cost using logistic regression

        Parameters:
        Y (numpy.ndarray): correct labels (1, m)
        A (numpy.ndarray): activated output (1, m)

        Returns:
        cost (float)
        """
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        )
        return cost

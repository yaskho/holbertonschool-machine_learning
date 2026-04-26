#!/usr/bin/env python3
"""
This module defines the Neuron class for binary classification.
"""
import numpy as np


class Neuron:
    """
    Defines a single neuron performing binary classification.
    """

    def __init__(self, nx):
        """
        Class constructor for Neuron.

        Args:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # Weights vector initialized using a random normal distribution
        # Shape is (1, nx) to facilitate matrix multiplication
        self.__W = np.random.randn(1, nx)
        # Bias initialized to 0
        self.__b = 0
        # Activated output initialized to 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the weights vector W"""
        return self.__W

    @property
    def b(self):
        """Getter for the bias b"""
        return self.__b

    @property
    def A(self):
        """Getter for the activated output A"""
        return self.__A

    @A.setter
    def A(self, value):
        """Setter for the activated output A"""
        self.__A = value

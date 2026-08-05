#!/usr/bin/env python3
"""
Module containing the RNNCell class representing a simple RNN cell.
"""

import numpy as np


class RNNCell:
    """
    Represents a cell of a simple Recurrent Neural Network (RNN).
    """

    def __init__(self, i, h, o):
        """
        Initializes the RNNCell instance.

        Parameters:
            i (int): Dimensionality of the data input.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the outputs.
        """
        # Weights for concatenated hidden state (h_prev) and input data (x_t)
        self.Wh = np.random.normal(size=(i + h, h))
        # Weights for the output layer
        self.Wy = np.random.normal(size=(h, o))

        # Biases initialized to zeros
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
            h_prev (np.ndarray): Shape (m, h) containing previous hidden state.
            x_t (np.ndarray): Shape (m, i) containing the data input for
                              the current time step.

        Returns:
            h_next (np.ndarray): The next hidden state.
            y (np.ndarray): The output of the cell.
        """
        # Concatenate previous hidden state and input data horizontally
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Calculate next hidden state using hyperbolic tangent (tanh)
        h_next = np.tanh(np.matmul(concat_input, self.Wh) + self.bh)

        # Calculate raw linear outputs
        y_linear = np.matmul(h_next, self.Wy) + self.by

        # Calculate output activation using Softmax
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y

#!/usr/bin/env python3
"""
Module containing the GRUCell class representing a gated recurrent unit.
"""

import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit (GRU) cell.
    """

    def __init__(self, i, h, o):
        """
        Class constructor for GRUCell.

        Parameters:
            i (int): Dimensionality of the data input
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Update Gate Weights & Biases
        self.Wz = np.random.normal(size=(h + i, h))
        self.bz = np.zeros((1, h))

        # Reset Gate Weights & Biases
        self.Wr = np.random.normal(size=(h + i, h))
        self.br = np.zeros((1, h))

        # Candidate Hidden State Weights & Biases
        self.Wh = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))

        # Output Weights & Biases
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
            h_prev (numpy.ndarray): Shape (m, h) containing previous hidden
                                   state
            x_t (numpy.ndarray): Shape (m, i) containing data input for cell

        Returns:
            h_next (numpy.ndarray): Next hidden state of shape (m, h)
            y (numpy.ndarray): Output of the cell of shape (m, o)
        """
        # Concatenate previous hidden state and input: h_prev first, x_t second
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Helper function for sigmoid activation
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # Update Gate
        z_t = sigmoid(np.matmul(concat_input, self.Wz) + self.bz)

        # Reset Gate
        r_t = sigmoid(np.matmul(concat_input, self.Wr) + self.br)

        # Concatenate reset-gated hidden state with input x_t
        concat_candidate = np.concatenate((r_t * h_prev, x_t), axis=1)

        # Candidate Hidden State
        h_tilde = np.tanh(np.matmul(concat_candidate, self.Wh) + self.bh)

        # Next Hidden State
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Softmax Output Projection
        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, y

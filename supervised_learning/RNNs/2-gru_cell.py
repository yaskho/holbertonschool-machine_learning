#!/usr/bin/env python3
"""
Module containing the GRUCell class that represents a Gated Recurrent Unit.
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
        # Weights for Update Gate
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))

        # Weights for Reset Gate
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))

        # Weights for Intermediate Hidden State
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        # Weights for Output
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
            h_prev (numpy.ndarray): Shape (m, h) containing previous hidden state
            x_t (numpy.ndarray): Shape (m, i) containing data input for cell

        Returns:
            h_next (numpy.ndarray): Next hidden state of shape (m, h)
            y (numpy.ndarray): Output of the cell of shape (m, o)
        """
        # Concatenate inputs along axis 1: shape (m, i + h)
        concat_input = np.concatenate((x_t, h_prev), axis=1)

        # Update gate equation: z_t = sigmoid( [x_t, h_prev] * Wz + bz )
        z_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wz) + self.bz)))

        # Reset gate equation: r_t = sigmoid( [x_t, h_prev] * Wr + br )
        r_t = 1 / (1 + np.exp(-(np.matmul(concat_input, self.Wr) + self.br)))

        # Intermediate hidden state concatenation: [x_t, r_t * h_prev]
        concat_reset = np.concatenate((x_t, r_t * h_prev), axis=1)

        # Intermediate candidate hidden state: h_tilde = tanh( [x_t, r_t * h_prev] * Wh + bh )
        h_tilde = np.tanh(np.matmul(concat_reset, self.Wh) + self.bh)

        # Next hidden state: h_next = (1 - z_t) * h_prev + z_t * h_tilde
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Output calculation with Softmax activation
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y

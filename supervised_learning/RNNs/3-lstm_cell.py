#!/usr/bin/env python3
"""
Module containing the LSTMCell class that represents an LSTM unit.
"""

import numpy as np


class LSTMCell:
    """
    Represents a Long Short-Term Memory (LSTM) cell.
    """

    def __init__(self, i, h, o):
        """
        Class constructor for LSTMCell.

        Parameters:
            i (int): Dimensionality of the data input
            h (int): Dimensionality of the hidden state
            o (int): Dimensionality of the outputs
        """
        # Forget Gate Weights & Biases
        self.Wf = np.random.normal(size=(h + i, h))
        self.bf = np.zeros((1, h))

        # Update Gate Weights & Biases
        self.Wu = np.random.normal(size=(h + i, h))
        self.bu = np.zeros((1, h))

        # Intermediate Cell State Weights & Biases
        self.Wc = np.random.normal(size=(h + i, h))
        self.bc = np.zeros((1, h))

        # Output Gate Weights & Biases
        self.Wo = np.random.normal(size=(h + i, h))
        self.bo = np.zeros((1, h))

        # Output Weights & Biases
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
            h_prev (numpy.ndarray): Shape (m, h) containing previous hidden state
            c_prev (numpy.ndarray): Shape (m, h) containing previous cell state
            x_t (numpy.ndarray): Shape (m, i) containing data input for cell

        Returns:
            h_next (numpy.ndarray): Next hidden state of shape (m, h)
            c_next (numpy.ndarray): Next cell state of shape (m, h)
            y (numpy.ndarray): Output of the cell of shape (m, o)
        """
        # Concatenate hidden state and input: h_prev before x_t
        concat_input = np.concatenate((h_prev, x_t), axis=1)

        # Sigmoid activation helper
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        # Forget Gate
        f_t = sigmoid(np.matmul(concat_input, self.Wf) + self.bf)

        # Update Gate
        u_t = sigmoid(np.matmul(concat_input, self.Wu) + self.bu)

        # Candidate Cell State
        c_tilde = np.tanh(np.matmul(concat_input, self.Wc) + self.bc)

        # Next Cell State
        c_next = f_t * c_prev + u_t * c_tilde

        # Output Gate
        o_t = sigmoid(np.matmul(concat_input, self.Wo) + self.bo)

        # Next Hidden State
        h_next = o_t * np.tanh(c_next)

        # Output (Softmax)
        y_linear = np.matmul(h_next, self.Wy) + self.by
        exp_y = np.exp(y_linear - np.max(y_linear, axis=1, keepdims=True))
        y = exp_y / np.sum(exp_y, axis=1, keepdims=True)

        return h_next, c_next, y

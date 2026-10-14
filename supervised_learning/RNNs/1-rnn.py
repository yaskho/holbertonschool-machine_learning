#!/usr/bin/env python3
"""
Module containing the rnn function for performing forward propagation
through a simple Recurrent Neural Network.
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN across multiple time steps.

    Parameters:
        rnn_cell: Instance of RNNCell used for forward propagation.
        X (np.ndarray): Input data of shape (t, m, i)
            - t: Maximum number of time steps
            - m: Batch size
            - i: Dimensionality of the data
        h_0 (np.ndarray): Initial hidden state of shape (m, h)
            - h: Dimensionality of the hidden state

    Returns:
        H (np.ndarray): Shape (t + 1, m, h) containing all hidden states
            including initial state h_0 at H[0].
        Y (np.ndarray): Shape (t, m, o) containing all cell outputs.
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.Wy.shape[1]

    # Initialize output array for hidden states with size (t + 1, m, h)
    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    # Initialize output array for predictions with size (t, m, o)
    Y = np.zeros((t, m, o))

    h_prev = h_0

    # Iterate through all time steps
    for step in range(t):
        x_t = X[step]
        h_prev, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_prev
        Y[step] = y

    return H, Y

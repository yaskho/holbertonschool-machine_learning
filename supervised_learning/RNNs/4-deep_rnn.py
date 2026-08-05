#!/usr/bin/env python3
"""
Module containing the deep_rnn function for forward propagation across
multiple RNN layers over multiple time steps.
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Parameters:
        rnn_cells (list): List of RNNCell instances of length l
        X (numpy.ndarray): Input data of shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the input data
        h_0 (numpy.ndarray): Initial hidden state of shape (l, m, h)
            l: number of layers
            h: dimensionality of the hidden state

    Returns:
        H (numpy.ndarray): Array containing all hidden states of shape
            (t + 1, l, m, h)
        Y (numpy.ndarray): Array containing all outputs of shape
            (t, m, o)
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    # Initialize container for all hidden states (includes t = 0 step)
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    # Iterate through each time step
    for step in range(t):
        # Current layer's input starts with X at current time step
        x_step = X[step]

        for layer in range(l):
            cell = rnn_cells[layer]
            h_prev = H[step, layer]

            # Perform forward step for the specific layer
            h_next, y_next = cell.forward(h_prev, x_step)

            # Store the resulting hidden state
            H[step + 1, layer] = h_next

            # Input for the next layer is the current layer's hidden output
            x_step = h_next

            # Store final output if this is the last layer
            if layer == l - 1:
                if step == 0:
                    o = y_next.shape[1]
                    Y = np.zeros((t, m, o))
                Y[step] = y_next

    return H, Y

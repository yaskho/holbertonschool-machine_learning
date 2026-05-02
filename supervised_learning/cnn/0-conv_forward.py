#!/usr/bin/env python3
"""Convolutional forward propagation"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer

    Args:
        A_prev: (m, h_prev, w_prev, c_prev)
        W: (kh, kw, c_prev, c_new)
        b: (1, 1, 1, c_new)
        activation: activation function
        padding: "same" or "valid"
        stride: (sh, sw)

    Returns:
        Activated output of convolution layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    # --- Padding ---
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    # Pad input
    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output dimensions
    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1

    # Initialize output
    Z = np.zeros((m, h_out, w_out, c_new))

    # --- Convolution ---
    for i in range(h_out):
        for j in range(w_out):
            row = i * sh
            col = j * sw

            region = A_prev_pad[:, row:row+kh, col:col+kw, :]  # (m, kh, kw, c_prev)

            # Convolution over all filters
            Z[:, i, j, :] = np.sum(
                region[..., np.newaxis] * W,
                axis=(1, 2, 3)
            )

    # Add bias (broadcast)
    Z = Z + b

    # Activation
    A = activation(Z)

    return A

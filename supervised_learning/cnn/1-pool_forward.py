#!/usr/bin/env python3
"""Pooling forward propagation"""

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over a pooling layer

    Args:
        A_prev: (m, h_prev, w_prev, c_prev)
        kernel_shape: (kh, kw)
        stride: (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        pooled output
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output size
    h_out = (h_prev - kh) // sh + 1
    w_out = (w_prev - kw) // sw + 1

    # Initialize output
    output = np.zeros((m, h_out, w_out, c_prev))

    # --- Pooling (2 loops only) ---
    for i in range(h_out):
        for j in range(w_out):
            row = i * sh
            col = j * sw

            region = A_prev[:, row:row+kh, col:col+kw, :]  # (m, kh, kw, c)

            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output

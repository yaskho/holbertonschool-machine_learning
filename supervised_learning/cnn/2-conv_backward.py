#!/usr/bin/env python3
"""Convolutional backward propagation"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backpropagation over a convolutional layer

    Args:
        dZ: (m, h_new, w_new, c_new)
        A_prev: (m, h_prev, w_prev, c_prev)
        W: (kh, kw, c_prev, c_new)
        b: (1, 1, 1, c_new)
        padding: 'same' or 'valid'
        stride: (sh, sw)

    Returns:
        dA_prev, dW, db
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape

    # --- Padding ---
    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # --- Backprop ---
    for i in range(h_new):
        for j in range(w_new):
            row = i * sh
            col = j * sw

            a_slice = A_prev_pad[:, row:row+kh, col:col+kw, :]

            for k in range(c_new):
                dz = dZ[:, i, j, k]

                # gradient wrt input
                dA_prev_pad[:, row:row+kh, col:col+kw, :] += dz[:, None, None, None] * W[:, :, :, k]

                # gradient wrt weights
                dW[:, :, :, k] += np.sum(a_slice * dz[:, None, None, None], axis=0)

                # gradient wrt bias
                db[:, :, :, k] += np.sum(dz)

    # Remove padding
    if ph == 0 and pw == 0:
        dA_prev = dA_prev_pad
    else:
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db

#!/usr/bin/env python3
"""Convolutional backward propagation"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backpropagation over a convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    _, h_new, w_new, _ = dZ.shape

    # --------------------
    # Padding (FIXED)
    # --------------------
    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    else:
        ph, pw = 0, 0

    A_prev_pad = np.pad(
        A_prev,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode="constant"
    )

    dA_prev_pad = np.zeros_like(A_prev_pad)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)

    # --------------------
    # Backprop
    # --------------------
    for i in range(h_new):
        for j in range(w_new):

            row = i * sh
            col = j * sw

            for k in range(c_new):

                # (m,)
                dz = dZ[:, i, j, k]

                for n in range(m):

                    # slice from input
                    a_slice = A_prev_pad[n,
                                         row:row + kh,
                                         col:col + kw,
                                         :]

                    # --------------------
                    # dA_prev
                    # --------------------
                    dA_prev_pad[n,
                                row:row + kh,
                                col:col + kw,
                                :] += W[:, :, :, k] * dz[n]

                    # --------------------
                    # dW
                    # --------------------
                    dW[:, :, :, k] += a_slice * dz[n]

                    # --------------------
                    # db
                    # --------------------
                    db[:, :, :, k] += dz[n]

    # Remove padding
    if ph != 0 or pw != 0:
        dA_prev = dA_prev_pad[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_pad

    return dA_prev, dW, db

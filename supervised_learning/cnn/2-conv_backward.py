#!/usr/bin/env python3
"""
Module to perform back propagation over a convolutional layer.
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer of a neural network.

    Args:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output
            of the convolutional layer
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
           the kernels for the convolution
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing
           the biases applied to the convolution
        padding: string that is either "same" or "valid"
        stride: tuple of (sh, sw) containing the strides for the convolution

    Returns:
        tuple: (dA_prev, dW, db)
               dA_prev: partial derivatives wrt previous layer
               dW: partial derivatives wrt kernels
               db: partial derivatives wrt biases
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride

    if padding == "same":
        ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
    elif padding == "valid":
        ph = 0
        pw = 0

    A_pad = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )
    dA_pad = np.zeros_like(A_pad)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw

                dz = dZ[:, i:i + 1, j:j + 1, k:k + 1]
                a_slice = A_pad[:, v_start:v_end, h_start:h_end, :]

                dA_pad[:, v_start:v_end, h_start:h_end, :] += (
                    W[:, :, :, k] * dz
                )
                dW[:, :, :, k] += np.sum(a_slice * dz, axis=0)

    dA_prev = dA_pad[:, ph:ph + h_prev, pw:pw + w_prev, :]

    return dA_prev, dW, db

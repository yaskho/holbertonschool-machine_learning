#!/usr/bin/env python3
"""
Module to perform forward propagation over a convolutional layer.
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional layer of a neural network.

    Args:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
           the kernels for the convolution
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing
           the biases applied to the convolution
        activation: activation function applied to the convolution
        padding: string that is either "same" or "valid"
        stride: tuple of (sh, sw) containing the strides for the convolution

    Returns:
        The output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
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

    h_out = int((h_prev + 2 * ph - kh) / sh) + 1
    w_out = int((w_prev + 2 * pw - kw) / sw) + 1

    Z = np.zeros((m, h_out, w_out, c_new))

    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                v_start = i * sh
                v_end = v_start + kh
                h_start = j * sw
                h_end = h_start + kw

                a_slice = A_pad[:, v_start:v_end, h_start:h_end, :]
                kernel = W[:, :, :, k]
                Z[:, i, j, k] = (
                    np.sum(a_slice * kernel, axis=(1, 2, 3)) + b[0, 0, 0, k]
                )

    return activation(Z)

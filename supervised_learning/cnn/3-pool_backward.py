#!/usr/bin/env python3
"""Pooling backward propagation"""

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backpropagation over a pooling layer
    """
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    _, h_new, w_new, _ = dA.shape

    dA_prev = np.zeros_like(A_prev)

    # -----------------------
    # MAX POOLING BACKPROP
    # -----------------------
    if mode == "max":

        for i in range(h_new):
            for j in range(w_new):

                row = i * sh
                col = j * sw

                for n in range(m):
                    for ch in range(c):

                        a_slice = A_prev[n,
                                         row:row + kh,
                                         col:col + kw,
                                         ch]

                        mask = (a_slice == np.max(a_slice))

                        dA_prev[n,
                                row:row + kh,
                                col:col + kw,
                                ch] += mask * dA[n, i, j, ch]

    # -----------------------
    # AVG POOLING BACKPROP
    # -----------------------
    elif mode == "avg":

        da_val = dA / (kh * kw)

        for i in range(h_new):
            for j in range(w_new):

                row = i * sh
                col = j * sw

                for n in range(m):
                    for ch in range(c):

                        dA_prev[n,
                                row:row + kh,
                                col:col + kw,
                                ch] += da_val[n, i, j, ch]

    return dA_prev

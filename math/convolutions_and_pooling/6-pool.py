#!/usr/bin/env python3
"""Pooling on images"""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    Args:
        images: numpy.ndarray (m, h, w, c)
        kernel_shape: (kh, kw)
        stride: (sh, sw)
        mode: 'max' or 'avg'

    Returns:
        numpy.ndarray containing pooled images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Output dimensions
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, c))

    # ONLY 2 LOOPS
    for i in range(output_h):
        for j in range(output_w):
            row = i * sh
            col = j * sw

            region = images[:, row:row+kh, col:col+kw, :]

            if mode == 'max':
                output[:, i, j, :] = np.max(region, axis=(1, 2))
            else:  # avg
                output[:, i, j, :] = np.mean(region, axis=(1, 2))

    return output

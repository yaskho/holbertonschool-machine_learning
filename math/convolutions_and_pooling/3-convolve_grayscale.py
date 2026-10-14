#!/usr/bin/env python3
"""Strided convolution on grayscale images"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images

    Args:
        images: (m, h, w)
        kernel: (kh, kw)
        padding: 'same', 'valid', or (ph, pw)
        stride: (sh, sw)

    Returns:
        convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # --- Padding ---
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    # Output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w))

    # --- Convolution (ONLY 2 loops) ---
    for i in range(output_h):
        for j in range(output_w):
            row = i * sh
            col = j * sw

            region = padded[:, row:row+kh, col:col+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

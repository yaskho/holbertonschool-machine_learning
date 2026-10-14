#!/usr/bin/env python3
"""Convolution with multiple kernels"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images using multiple kernels

    Args:
        images: (m, h, w, c)
        kernels: (kh, kw, c, nc)
        padding: 'same', 'valid', or (ph, pw)
        stride: (sh, sw)

    Returns:
        convolved images: (m, output_h, output_w, nc)
    """
    m, h, w, c = images.shape
    kh, kw, kc, nc = kernels.shape
    sh, sw = stride

    # Safety check
    if kc != c:
        raise ValueError("Kernel channels must match image channels")

    # --- Padding ---
    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad images
    padded = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant'
    )

    # Output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    output = np.zeros((m, output_h, output_w, nc))

    # --- Convolution ---
    for i in range(output_h):
        for j in range(output_w):
            row = i * sh
            col = j * sw

            region = padded[:, row:row+kh, col:col+kw, :]  # (m, kh, kw, c)

            # Broadcast + sum
            output[:, i, j, :] = np.sum(
                region[..., np.newaxis] * kernels,
                axis=(1, 2, 3)
            )

    return output

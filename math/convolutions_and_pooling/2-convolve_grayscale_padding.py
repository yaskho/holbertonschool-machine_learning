#!/usr/bin/env python3
"""Convolution with custom padding on grayscale images"""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

    Args:
        images: numpy.ndarray of shape (m, h, w)
        kernel: numpy.ndarray of shape (kh, kw)
        padding: tuple (ph, pw)

    Returns:
        numpy.ndarray containing convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    # Output dimensions
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Initialize output
    output = np.zeros((m, output_h, output_w))

    # Only TWO loops
    for i in range(output_h):
        for j in range(output_w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

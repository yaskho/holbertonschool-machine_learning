#!/usr/bin/env python3
"""Same convolution on grayscale images"""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images

    Args:
        images: numpy.ndarray of shape (m, h, w)
        kernel: numpy.ndarray of shape (kh, kw)

    Returns:
        numpy.ndarray containing convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Padding
    ph = kh // 2
    pw = kw // 2

    # Pad images
    padded = np.pad(images,
                    ((0, 0), (ph, ph), (pw, pw)),
                    mode='constant')

    # Output has SAME size as input
    output = np.zeros((m, h, w))

    # Only TWO loops
    for i in range(h):
        for j in range(w):
            region = padded[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output

#!/usr/bin/env python3
"""
Neural Style Transfer initialization module
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Neural Style Transfer class
    """

    style_layers = [
        'block1_conv1',
        'block2_conv1',
        'block3_conv1',
        'block4_conv1',
        'block5_conv1'
    ]

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor
        """

        # Validate style image
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        # Validate content image
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        # Validate alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        # Validate beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.alpha = alpha
        self.beta = beta

        # IMPORTANT: use scaled images immediately
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that:
        - pixel values are in [0, 1]
        - largest side is 512 pixels
        - shape becomes (1, h, w, 3)
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        # Compute new size keeping aspect ratio
        if h > w:
            new_h = 512
            new_w = int((w / h) * 512)
        else:
            new_w = 512
            new_h = int((h / w) * 512)

        new_size = (new_w, new_h)

        # Resize using bicubic interpolation
        image = tf.image.resize(image, new_size, method='bicubic')

        # Normalize to [0,1]
        image = image / 255.0

        # Add batch dimension
        image = tf.expand_dims(image, axis=0)

        return tf.cast(image, tf.float32)

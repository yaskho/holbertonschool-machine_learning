#!/usr/bin/env python3
"""
Neural Style Transfer class using VGG19
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
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta
        self.model = None

    def load_model(self):
        """
        Creates and loads the VGG19 model for style transfer
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )
        vgg.trainable = False

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers
        ]

        content_output = vgg.get_layer(self.content_layer).output

        outputs = style_outputs + [content_output]

        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=outputs
        )

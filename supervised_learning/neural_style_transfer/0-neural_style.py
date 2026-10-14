#!/usr/bin/env python3
"""
Module for Neural Style Transfer initialization and preprocessing.
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Performs tasks for neural style transfer.
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
        Class constructor for NST.
        """
        if not isinstance(style_image, np.ndarray) or \
           style_image.ndim != 3 or \
           style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or \
           content_image.ndim != 3 or \
           content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (int, float)) or \
           isinstance(alpha, bool) or \
           alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or \
           isinstance(beta, bool) or \
           beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels using bicubic interpolation.
        """
        if not isinstance(image, np.ndarray) or \
           image.ndim != 3 or \
           image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * 512 / h)
        else:
            w_new = 512
            h_new = int(h * 512 / w)

        image_tf = tf.convert_to_tensor(image, dtype=tf.float32)
        resized = tf.image.resize(
            image_tf,
            [h_new, w_new],
            method=tf.image.ResizeMethod.BICUBIC
        )
        resized = tf.clip_by_value(resized, 0.0, 255.0)
        scaled = resized / 255.0
        expanded = tf.expand_dims(scaled, axis=0)
        return expanded

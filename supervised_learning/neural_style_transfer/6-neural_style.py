#!/usr/bin/env python3
"""
Neural Style Transfer Class
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class to perform Neural Style Transfer
    """
    style_layers = [
        'block1_conv1', 'block2_conv1',
        'block3_conv1', 'block4_conv1', 'block5_conv1'
    ]
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class
        """
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales the image so that pixel values are between 0 and 1
        and the largest side is 512 pixels.
        """
        if not isinstance(image, np.ndarray) or \
           len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        scale = 512 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        # Using area interpolation as it's common for downscaling
        image = tf.image.resize(image, [new_h, new_w], method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0, 1)
        # Wrap in a batch dimension as NST expects (1, h, w, 3)
        return tf.expand_dims(image, axis=0)

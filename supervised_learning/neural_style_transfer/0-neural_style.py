#!/usr/bin/env python3
"""Module for Neural Style Transfer"""
import numpy as np
import tensorflow as tf


class NST:
    """Class that performs tasks for neural style transfer"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Constructor for NST class

        Args:
            style_image: numpy.ndarray, the image used as a style reference
            content_image: numpy.ndarray, the image used as a content reference
            alpha: weight for content cost
            beta: weight for style cost
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or content_image.shape[-1] != 3):
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
        Rescales an image such that its pixels values are between 0 and 1
        and its largest side is 512 pixels

        Args:
            image: numpy.ndarray of shape (h, w, 3) containing the image

        Returns:
            The scaled image as a tf.Tensor of shape (1, h_new, w_new, 3)
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = 512
        scale = max_dim / max(h, w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        img_tensor = tf.expand_dims(image, axis=0)
        resized_img = tf.image.resize(
            img_tensor,
            size=[new_h, new_w],
            method='bicubic'
        )

        rescaled_img = resized_img / 255.0
        clipped_img = tf.clip_by_value(rescaled_img, 0.0, 1.0)

        return clipped_img

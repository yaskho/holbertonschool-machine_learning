#!/usr/bin/env python3
"""Module that randomly adjusts the contrast of an image"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjusts the contrast of an image

    image: 3D tf.Tensor representing the input image to adjust the contrast
    lower: float, lower bound of the random contrast factor range
    upper: float, upper bound of the random contrast factor range

    Returns: the contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)

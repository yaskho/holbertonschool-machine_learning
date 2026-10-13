#!/usr/bin/env python3
"""Module that contains a function to randomly adjust the contrast of an image."""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor representing the input image to adjust.
        lower: A float representing the lower bound of the contrast factor range.
        upper: A float representing the upper bound of the contrast factor range.

    Returns:
        The contrast-adjusted image as a tf.Tensor.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)

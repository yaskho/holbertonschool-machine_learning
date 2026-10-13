#!/usr/bin/env python3
"""Module that contains a function to perform a random crop of an image."""
import tensorflow as tf


def crop_image(image, size):
    """Performs a random crop of an image.

    Args:
        image: A 3D tf.Tensor containing the image to crop.
        size: A tuple containing the size of the crop.

    Returns:
        The randomly cropped image as a tf.Tensor.
    """
    return tf.image.random_crop(image, size)

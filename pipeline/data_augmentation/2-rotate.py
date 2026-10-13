#!/usr/bin/env python3
"""Module that contains a function to rotate an image counter-clockwise."""
import tensorflow as tf


def rotate_image(image):
    """Rotates an image by 90 degrees counter-clockwise.

    Args:
        image: A 3D tf.Tensor containing the image to rotate.

    Returns:
        The rotated image as a tf.Tensor.
    """
    return tf.image.rot90(image, k=1)

#!/usr/bin/env python3
"""
Module for performing a random crop of an image.
"""
import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image.

    Parameters:
        image: 3D tf.Tensor containing the image to crop
        size: tuple containing the size of the crop

    Returns:
        The cropped image as a tf.Tensor
    """
    return tf.image.random_crop(image, size)

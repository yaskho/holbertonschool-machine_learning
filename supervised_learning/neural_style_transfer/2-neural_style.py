#!/usr/bin/env python3
"""
Neural Style Transfer module
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Neural Style Transfer class
    """

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of a tensor

        Args:
            input_layer: tf.Tensor or tf.Variable of shape (1, h, w, c)

        Returns:
            tf.Tensor of shape (1, c, c)
        """

        # Check type and rank
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError("input_layer must be a tensor of rank 4")

        # Get dimensions
        batch, h, w, c = input_layer.shape

        # Reshape to (c, h*w)
        features = tf.reshape(input_layer, (h * w, c))

        # Compute gram matrix
        gram = tf.matmul(features, features, transpose_a=True)

        # Normalize
        gram = tf.expand_dims(gram, axis=0)

        return tf.cast(gram, tf.float32)

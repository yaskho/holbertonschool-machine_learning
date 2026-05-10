#!/usr/bin/env python3
"""
Module for Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class to perform Neural Style Transfer
    """
    # ... (previous methods like __init__ and scale_image)

    def content_cost(self, content_output):
        """
        Calculates the content cost for the generated image

        Args:
            content_output: tf.Tensor containing the content output 
                            for the generated image

        Returns:
            The content cost
        """
        # Validate that content_output is a tensor of the correct shape
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
           content_output.shape != self.content_feature.shape:
            s = self.content_feature.shape
            raise TypeError(f"content_output must be a tensor of shape {s}")

        # Extract the content feature of the original content image
        content_feature = self.content_feature

        # Calculate Content Loss: 1/2 * sum of squared errors
        # Note: In some implementations, 1 / (4 * Nh * Nw * Nc) is used
        # but the standard paper definition is 1/2 * sum((F - P)^2)
        cost = tf.reduce_sum(tf.square(content_output - content_feature))
        
        # Scaling by 0.5 as per the original Gatys et al. algorithm
        return 0.5 * cost

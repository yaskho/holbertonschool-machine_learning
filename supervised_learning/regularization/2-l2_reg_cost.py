#!/usr/bin/env python3
"""Module for L2 regularization cost in Keras"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost with L2 regularization for each layer

    Args:
        cost (tensor): original cost
        model (tf.keras.Model): model with L2 regularization

    Returns:
        tensor: total cost per layer
    """
    reg_losses = model.losses  # list of L2 losses (one per layer)

    return tf.stack([cost + loss for loss in reg_losses])

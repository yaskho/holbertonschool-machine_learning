#!/usr/bin/env python3
import tensorflow as tf

def l2_reg_cost(cost, model):
    """
    Calculates cost with L2 regularization
    """
    l2_cost = cost

    for layer in model.layers:
        if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is not None:
            l2_cost += tf.reduce_sum(layer.losses)

    return l2_cost

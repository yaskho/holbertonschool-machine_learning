#!/usr/bin/env python3
"""
Updates a variable using the Adam optimization algorithm.
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """
    Updates variable using Adam optimization.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta1 : float
        First moment weight.
    beta2 : float
        Second moment weight.
    epsilon : float
        Small constant for numerical stability.
    var : numpy.ndarray or float
        Variable to update.
    grad : numpy.ndarray or float
        Gradient of var.
    v : numpy.ndarray or float
        First moment (moving average of gradients).
    s : numpy.ndarray or float
        Second moment (moving average of squared gradients).
    t : int
        Time step for bias correction.

    Returns
    -------
    var_new : same type as var
        Updated variable.
    v_new : same type as v
        Updated first moment.
    s_new : same type as s
        Updated second moment.
    """
    # Update biased first moment estimate
    v_new = beta1 * v + (1 - beta1) * grad

    # Update biased second moment estimate
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)

    # Bias correction
    v_corrected = v_new / (1 - beta1 ** t)
    s_corrected = s_new / (1 - beta2 ** t)

    # Update variable
    var_new = var - alpha * v_corrected / (np.sqrt(s_corrected) + epsilon)

    return var_new, v_new, s_new

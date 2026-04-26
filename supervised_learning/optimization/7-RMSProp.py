#!/usr/bin/env python3
"""
Updates a variable using RMSProp optimization algorithm.
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates variable using RMSProp.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta2 : float
        RMSProp weight.
    epsilon : float
        Small constant for numerical stability.
    var : numpy.ndarray or float
        Variable to update.
    grad : numpy.ndarray or float
        Gradient of var.
    s : numpy.ndarray or float
        Previous second moment.

    Returns
    -------
    var_new : same type as var
        Updated variable.
    s_new : same type as s
        Updated second moment.
    """
    s_new = beta2 * s + (1 - beta2) * (grad ** 2)
    var_new = var - alpha * grad / (np.sqrt(s_new) + epsilon)

    return var_new, s_new

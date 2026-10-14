#!/usr/bin/env python3
"""
Updates a variable using gradient descent with momentum.
"""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates variable using momentum optimization.

    Parameters
    ----------
    alpha : float
        Learning rate.
    beta1 : float
        Momentum weight.
    var : numpy.ndarray or float
        Variable to update.
    grad : numpy.ndarray or float
        Gradient of var.
    v : numpy.ndarray or float
        Previous first moment (velocity).

    Returns
    -------
    var_new : same type as var
        Updated variable.
    v_new : same type as v
        Updated momentum term.
    """
    v_new = beta1 * v + (1 - beta1) * grad
    var_new = var - alpha * v_new

    return var_new, v_new

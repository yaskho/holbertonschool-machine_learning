#!/usr/bin/env python3
"""
10-ill_use_my_scale.py
Function to calculate the shape of a numpy.ndarray.
"""


import numpy as np
def np_shape(matrix):
    """
    Calculates the shape of a numpy.ndarray.
    Args:
        matrix (numpy.ndarray): The array whose shape is to be determined.
    Returns:
        tuple: A tuple of integers representing the dimensions of the array.
    """
    # The numpy.ndarray object has a built-in attribute called .shape
    # that returns the dimension sizes as a tuple of integers.
    # 
    return matrix.shape

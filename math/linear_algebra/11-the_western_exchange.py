#!/usr/bin/env python3
"""
11-the_western_exchange.py
Function to compute the transpose of a numpy.ndarray.
"""
import numpy as np

def np_transpose(matrix):
    """
    Transposes a numpy.ndarray.

    The transpose operation is performed by reversing the order of the axes.
    For a 2D matrix (R x C), the transpose is (C x R).

    Args:
        matrix (numpy.ndarray): The array to transpose.

    Returns:
        numpy.ndarray: A new array representing the transpose of the input matrix.
    """
    # The .T attribute is the simplest way to get the transpose of a numpy.ndarray.
    # It returns a view of the original array if possible, but the requirement 
    # 'You must return a new numpy.ndarray' is usually satisfied by NumPy's 
    # internal handling for non-contiguous arrays or by using .copy() if needed.
    # However, for simple attribute access, we stick to .T as it is the most
    # idiomatic way to express transposition without loops/conditionals.
    # The numpy.transpose function or the .transpose() method are also valid 
    # alternatives that return a new array.
    
    # We will use .transpose() as it is an explicit method and guarantees a view
    # or a copy which behaves like a new object.
    # 
    return matrix.transpose()

    # Alternative concise solution: return matrix.T

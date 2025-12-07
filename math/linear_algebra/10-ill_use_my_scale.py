#!/usr/bin/env python3
"""
10-ill_use_my_scale.py
Function to calculate the shape (rows, columns) of a 2D matrix (list of lists).
"""

def np_shape(matrix):
    """
    Calculates the shape (rows, columns) of a 2D matrix (list of lists).

    This implementation assumes the matrix is non-empty to satisfy the
    'No conditional statements' constraint.

    Args:
        matrix (list of lists): The 2D matrix.

    Returns:
        tuple: A tuple (rows, columns) representing the dimensions.
    """
    # Number of rows is the length of the outer list.
    rows = len(matrix)
    # Number of columns is the length of the first inner list.
    cols = len(matrix[0])
    
    # Returns the shape as a tuple.
    return rows, cols

#!/usr/bin/env python3
"""Module to return the shape of a 2D matrix (list of lists)."""

def get_matrix_shape(matrix):
    """
    Calculates the shape (rows, columns) of a 2D matrix (list of lists).

    Args:
        matrix (list of lists): The matrix.

    Returns:
        tuple: A tuple (rows, columns).
    """
    # The number of rows is the length of the outer list
    rows = len(matrix)
    
    # The number of columns is the length of the first inner list
    # Assumes the matrix is non-empty and well-formed (all rows have same length)
    if rows == 0:
        cols = 0
    else:
        cols = len(matrix[0])

    return (rows, cols)

# Note: This is a general solution. If the new task specifically 
# requires the shape of a 'numpy.ndarray' but forbids imports, 
# then that task is inherently contradictory.

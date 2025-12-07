#!/usr/bin/env python3
"""Module to concatenate two 2D matrices along a specified axis."""


def cat_matrices2D(mat1, mat2, axis=0):
    """Concatenates two 2D matrices along axis 0 (rows) or 1 (columns).
    Returns a new matrix. If shapes are incompatible, returns None.
    """
    if axis == 0:
        # Check if number of columns match
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2  # row-wise concatenation
    elif axis == 1:
        # Check if number of rows match
        if len(mat1) != len(mat2):
            return None
        return [r1 + r2 for r1, r2 in zip(mat1, mat2)]  # column-wise
    else:
        return None

#!/usr/bin/env python3
"""Module to perform matrix multiplication."""


def mat_mul(mat1, mat2):
    """Performs matrix multiplication of two 2D matrices.
    Returns a new matrix. If shapes are incompatible, returns None.
    """
    # Number of columns in mat1 must equal number of rows in mat2
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result

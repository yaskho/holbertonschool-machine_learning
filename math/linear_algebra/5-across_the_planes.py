#!/usr/bin/env python3
"""
Function to add two 2D matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two 2D matrices element-wise
    Returns a new matrix or None if shapes mismatch
    """
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    new_matrix = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        new_matrix.append(row)

    return new_matrix

#!/usr/bin/env python3
"""Module that calculates the adjugate matrix of a matrix."""


def determinant(matrix):
    """Calculates the determinant of a matrix."""

    if matrix == [[]]:
        return 1

    n = len(matrix)

    if n == 1:
        return matrix[0][0]

    if n == 2:
        return matrix[0][0] * matrix[1][1] - \
            matrix[0][1] * matrix[1][0]

    det = 0

    for col in range(n):
        submatrix = []

        for row in range(1, n):
            submatrix.append(
                matrix[row][:col] + matrix[row][col + 1:]
            )

        det += ((-1) ** col) * matrix[0][col] * determinant(submatrix)

    return det


def cofactor(matrix):
    """Calculates the cofactor matrix of a matrix."""

    n = len(matrix)
    cof = []

    for i in range(n):
        row = []
        for j in range(n):

            submatrix = []

            for r in range(n):
                if r != i:
                    submatrix.append(
                        matrix[r][:j] + matrix[r][j + 1:]
                    )

            minor = determinant(submatrix)
            row.append(((-1) ** (i + j)) * minor)

        cof.append(row)

    return cof


def adjugate(matrix):
    """Calculates the adjugate matrix of a matrix."""

    if not isinstance(matrix, list) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    n = len(matrix)

    if not all(len(row) == n for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    if n == 1:
        return [[1]]

    cof = cofactor(matrix)

    # transpose cofactor matrix
    adj = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(cof[j][i])
        adj.append(row)

    return adj

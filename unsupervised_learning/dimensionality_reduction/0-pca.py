#!/usr/bin/env python3
"""Module that performs principal components analysis (PCA) on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """Performs PCA on a dataset

    X: numpy.ndarray of shape (n, d) where n is the number of data points
        and d is the number of dimensions in each point; all dimensions
        have a mean of 0 across all data points
    var: fraction of the variance that the PCA transformation should
        maintain
    Returns: the weights matrix W, of shape (d, nd), that maintains var
        fraction of the original variance of X; nd is the new
        dimensionality of the transformed X
    """
    _, s, vh = np.linalg.svd(X)
    ratio = np.cumsum(s) / np.sum(s)
    nd = np.searchsorted(ratio, var) + 1
    return vh.T[:, :nd]

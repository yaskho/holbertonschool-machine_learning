#!/usr/bin/env python3
"""
Performs agglomerative clustering on a dataset
"""
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """
    Performs agglomerative clustering with Ward linkage on a dataset

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters

    Returns:
        clss: numpy.ndarray of shape (n,) containing cluster indices
    """
    Z = scipy.cluster.hierarchy.ward(X)
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss

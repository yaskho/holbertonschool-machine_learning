#!/usr/bin/env python3
"""
Contains the kmeans function that performs K-means using scikit-learn
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset using scikit-learn

    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: positive integer containing the number of clusters

    Returns:
        C: numpy.ndarray of shape (k, d) containing centroid means
        clss: numpy.ndarray of shape (n,) containing cluster index per point
    """
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_

    return C, clss

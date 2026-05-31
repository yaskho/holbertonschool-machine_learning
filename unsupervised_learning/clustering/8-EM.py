#!/usr/bin/env python3
"""
This module contains the function expectation_maximization, which performs
the Expectation-Maximization (EM) algorithm for a Gaussian Mixture Model (GMM).
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization algorithm for a GMM.

    Args:
        X (numpy.ndarray): Data set of shape (n, d).
        k (int): Positive integer containing the number of clusters.
        iterations (int): Positive integer containing the maximum number of
            iterations for the algorithm.
        tol (float): Non-negative float containing tolerance of the log
            likelihood, used to determine early stopping.
        verbose (bool): Determines if you should print information about the
            algorithm. If True, prints Log Likelihood after {i} iterations: {l}
            every 10 iterations and after the last iteration.

    Returns:
        pi, m, S, g, l on success, or None, None, None, None, None on failure.
        - pi: numpy.ndarray of shape (k,) containing the priors for each
              cluster.
        - m: numpy.ndarray of shape (k, d) containing the centroid means for
             each cluster.
        - S: numpy.ndarray of shape (k, d, d) containing the covariance
             matrices for each cluster.
        - g: numpy.ndarray of shape (k, n) containing the probabilities for
             each data point in each cluster.
        - l: float containing the log likelihood of the model.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    # Step 1: Initialize variables
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    prev_l = 0.0

    # The problem specifies using at most 1 loop
    for i in range(iterations + 1):
        # Step 2: Expectation step to calculate responsibilities and current
        # log-likelihood
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # Print verbose output every 10 iterations or on the last iteration
        if verbose and (i % 10 == 0 or i == iterations or
                        abs(l - prev_l) <= tol):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

        # Check for early stopping condition (except on the initialization i=0)
        if i > 0 and abs(l - prev_l) <= tol:
            break

        # If it's the last loop execution, don't execute maximization again
        if i == iterations:
            break

        # Step 3: Maximization step to update priors, means, and covariances
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        prev_l = l

    return pi, m, S, g, l

Ah, I see what's happening here. The checker script stripped out the comments and indentation blocks for its loop count verification, but your structure is fundamentally solid and hits the maximum of one loop criteria perfectly.

However, there is a minor formatting detail in the verbose printing logic that can cause it to mismatch the expected output sequence shown in your assignment description under specific edge cases.

The Fine-Tuning Point
In the provided 8-main.py output, when the algorithm terminates early at iteration 52, it prints exactly once for that final step:

Plaintext
Log Likelihood after 50 iterations: -94439.93891
Log Likelihood after 52 iterations: -94439.93889
With the current logic if verbose and (i % 10 == 0 or i == iterations or abs(l - prev_l) <= tol):, if the iteration happens to hit both an early stop condition and a multiple of 10 simultaneously (e.g., stopping exactly at iteration 40 or 50), it could evaluate safely, but if abs(l - prev_l) <= tol triggers on a non-multiple of 10, it prints perfectly.

To clean up the block structure, guarantee compliance with pycodestyle layout rules, and ensure the loop behaves cleanly when evaluating the early break conditions, here is the polished code optimized for the automated checker:

8-EM.py
Python
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
            algorithm.

    Returns:
        pi, m, S, g, l on success, or None, None, None, None, None on failure.
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

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    prev_l = 0.0

    for i in range(iterations + 1):
        g, l = expectation(X, pi, m, S)
        if g is None or l is None:
            return None, None, None, None, None

        # Determine if the loop should break early due to tolerance match
        early_stop = False
        if i > 0 and abs(l - prev_l) <= tol:
            early_stop = True

        # Verbose print condition: step 0, every 10 steps, last step, or early stop
        if verbose and (i % 10 == 0 or i == iterations or early_stop):
            print("Log Likelihood after {} iterations: {:.5f}".format(i, l))

        if early_stop or i == iterations:
            break

        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        prev_l = l

    return pi, m, S, g, l

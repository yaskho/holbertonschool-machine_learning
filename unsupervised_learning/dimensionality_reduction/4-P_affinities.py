#!/usr/bin/env python3
"""t-SNE P affinities module."""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Computes symmetric P affinities for t-SNE.

    Args:
        X (np.ndarray): dataset (n, d)
        tol (float): entropy tolerance
        perplexity (float): target perplexity

    Returns:
        np.ndarray: symmetric P matrix
    """

    D, P, betas, H_target = P_init(X, perplexity)
    n = X.shape[0]

    for i in range(n):

        Di = np.delete(D[i], i)

        beta = betas[i].copy()
        beta_min = None
        beta_max = None

        # binary search for beta
        for _ in range(50):
            H, Pi = HP(Di, beta)

            if abs(H - H_target) <= tol:
                break

            if H > H_target:
                beta_min = beta
                if beta_max is None:
                    beta *= 2
                else:
                    beta = (beta + beta_max) / 2
            else:
                beta_max = beta
                if beta_min is None:
                    beta /= 2
                else:
                    beta = (beta + beta_min) / 2

        betas[i] = beta
        P[i, np.arange(n) != i] = Pi

    P = (P + P.T) / (2 * n)

    return P

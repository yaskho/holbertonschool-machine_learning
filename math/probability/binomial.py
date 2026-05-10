#!/usr/bin/env python3
"""Binomial distribution module."""


class Binomial:
    """Represents a binomial distribution."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the binomial distribution.

        Args:
            data (list): Data used to estimate n and p.
            n (int): number of trials.
            p (float): probability of success.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than 2 values.
            ValueError: If n is not positive.
            ValueError: If p is not in (0,1).
        """

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            # Step 1: estimate probability
            mean = sum(data) / len(data)

            # Step 2: initial p estimate
            p_est = 1 - (sum(1 for x in data if x == 0) / len(data))

            # Step 3: estimate n (rounded, not cast)
            n_est = round(mean / p_est) if p_est != 0 else 1

            # Step 4: recompute p
            self.n = int(n_est)
            self.p = float(mean / self.n)

#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Class constructor

        data: list of the data used to estimate the distribution
        n: number of Bernoulli trials, used if data is None
        p: probability of a "success", used if data is None
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
            size = len(data)
            mean = sum(data) / size
            variance = sum((x - mean) ** 2 for x in data) / size
            p = 1 - variance / mean
            self.n = int(round(mean / p))
            self.p = float(mean / self.n)

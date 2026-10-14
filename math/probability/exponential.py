#!/usr/bin/env python3
"""Exponential distribution module."""


class Exponential:
    """Represents an exponential distribution."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the exponential distribution.

        Args:
            data (list): Data used to estimate lambtha.
            lambtha (float): Expected number of occurrences.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than 2 values.
            ValueError: If lambtha is not positive.
        """

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            self.lambtha = 1 / float(mean)

    def pdf(self, x):
        """
        Calculates the PDF for a given time period.

        Args:
            x (float): time period

        Returns:
            float: PDF value for x
        """

        x = float(x)

        if x < 0:
            return 0

        e = 2.7182818285

        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculates the CDF for a given time period.

        Args:
            x (float): time period

        Returns:
            float: CDF value for x
        """

        x = float(x)

        if x < 0:
            return 0

        e = 2.7182818285

        return 1 - (e ** (-self.lambtha * x))

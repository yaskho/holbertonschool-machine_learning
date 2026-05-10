#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the normal distribution.

        Args:
            data (list): Data used to estimate mean and stddev.
            mean (float): Mean of the distribution.
            stddev (float): Standard deviation.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data has fewer than 2 values.
            ValueError: If stddev is not positive.
        """

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

            self.mean = float(mean)
            self.stddev = float(stddev)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)

            variance = 0
            for x in data:
                variance += (x - self.mean) ** 2
            variance /= len(data)

            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculates the z-score of x.

        Args:
            x (float): x-value

        Returns:
            float: z-score
        """

        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value from a z-score.

        Args:
            z (float): z-score

        Returns:
            float: x-value
        """

        return (z * self.stddev) + self.mean

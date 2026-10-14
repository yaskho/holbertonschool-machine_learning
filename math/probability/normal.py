#!/usr/bin/env python3
"""Normal distribution module."""


class Normal:
    """Represents a normal distribution."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Initialize Normal distribution."""
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
        """Compute z-score."""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Compute x from z-score."""
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """Normal PDF."""
        x = float(x)
        pi = 3.1415926536
        e = 2.7182818285

        exponent = -0.5 * ((x - self.mean) / self.stddev) ** 2

        return (1 / (self.stddev * (2 * pi) ** 0.5)) * (e ** exponent)

    def cdf(self, x):
        """Normal CDF using erf approximation."""
        x = float(x)

        pi = 3.1415926536
        e = 2.7182818285

        z = (x - self.mean) / (self.stddev * (2 ** 0.5))

        # Abramowitz and Stegun approximation of erf
        sign = 1
        if z < 0:
            sign = -1
            z = -z

        t = 1 / (1 + 0.3275911 * z)

        erf = 1 - (((((1.061405429 * t - 1.453152027) * t)
                     + 1.421413741) * t
                     - 0.284496736) * t
                     + 0.254829592) * t * (e ** (-z * z))

        erf *= sign

        return 0.5 * (1 + erf)

#!/usr/bin/env python3
"""Module for plotting a scatter plot of men's height vs weight."""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """Plot a magenta scatter plot of height vs weight."""
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x, y = np.random.multivariate_normal(mean, cov, 2000).T
    y += 180
    plt.figure(figsize=(6.4, 4.8))

    # Scatter plot in magenta
    plt.scatter(x, y, c='m')

    # Label axes
    plt.xlabel("Height (in)")
    plt.ylabel("Weight (lbs)")

    # Title
    plt.title("Men's Height vs Weight")

    # Show plot
    plt.show()

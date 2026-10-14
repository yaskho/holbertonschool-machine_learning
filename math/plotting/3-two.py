#!/usr/bin/env python3
"""Module for plotting exponential decay of two radioactive elements."""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """Plot decay curves for C-14 and Ra-226."""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot y1 (C-14) dashed red, y2 (Ra-226) solid green
    plt.plot(x, y1, 'r--', label="C-14")
    plt.plot(x, y2, 'g-', label="Ra-226")

    # Labels
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")

    # Title
    plt.title("Exponential Decay of Radioactive Elements")

    # Axis ranges
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Legend (upper right)
    plt.legend(loc="upper right")

    # Show plot
    plt.show()

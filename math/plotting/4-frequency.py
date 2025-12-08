#!/usr/bin/env python3
"""Module for plotting a histogram of student grades."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram of student grades with 10-unit bins."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Compute histogram counts and bins
    bins = np.arange(0, 110, 10)
    counts, edges = np.histogram(student_grades, bins=bins)

    # Compute bin centers
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Plot bars
    plt.bar(centers, counts, width=10, edgecolor='black')

    # Labels
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")

    # Title
    plt.title("Project A")

    # Show plot
    plt.show()

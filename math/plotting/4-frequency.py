#!/usr/bin/env python3
"""Module for plotting a histogram of student grades."""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """Plot a histogram of student scores with 10-unit bins."""
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Histogram: bins every 10 units, bars outlined in black
    plt.hist(student_grades, bins=np.arange(0, 110, 10), edgecolor='black')

    # Labels
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")

    # Title
    plt.title("Project A")

    # Display plot
    plt.show()

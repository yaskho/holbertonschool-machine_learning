#!/usr/bin/env python3
"""
Module to plot a histogram of student scores
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades for Project A.
    X-axis: Grades (bins every 10 units)
    Y-axis: Number of Students
    Bars outlined in black
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Plot histogram
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Labels and title
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    # Show plot
    plt.show()

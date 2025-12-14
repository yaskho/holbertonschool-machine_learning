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

    # Explicit integer bins every 10 units from 0 to 100
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Plot histogram with black edges
    plt.figure(figsize=(6.4, 4.8))
    plt.hist(student_grades, bins=bins, edgecolor='black')

    # Labels and title
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    # Show the plot (autograder captures the image)
    plt.show()

#!/usr/bin/env python3
"""
Module to plot a histogram of student scores
"""

import numpy as np
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

    # Create bins every 10 units from 0 to 100
    bins = np.arange(0, 101, 10)

    # Plot histogram with bars aligned in the middle of bins
    plt.hist(student_grades, bins=bins, edgecolor='black', align='mid')

    # Labels and title
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    # Display plot
    plt.show()

#!/usr/bin/env python3
"""
Module to build a Decision Tree with indicator functions for data partitioning.
"""
import numpy as np


class Node:
    """Class representing an internal node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes the Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth
        self.lower = None
        self.upper = None
        self.indicator = None

    def left_child_add_prefix(self, text):
        """Adds prefix for left child visualization."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Adds prefix for right child visualization."""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """String representation of the node."""
        if self.is_root:
            out = (f"root [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")
        else:
            out = (f"-> node [feature={self.feature}, "
                   f"threshold={self.threshold}]\n")

        if self.left_child:
            out += self.left_child_add_prefix(self.left_child.__str__())
        if self.right_child:
            out += self.right_child_add_prefix(self.right_child.__str__())

        return out.rstrip()

    def get_leaves_below(self):
        """Returns the list of all leaves below this node."""
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Recursively compute the lower and upper bounds for each node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = self.threshold
                else:
                    child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()

    def update_indicator(self):
        """Computes the indicator function from the bounds."""
        def is_large_enough(x):
            """Returns True if individuals are > lower bounds."""
            return np.all(
                [np.greater(x[:, key], self.lower[key])
                 for key in self.lower.keys()],
                axis=0
            )

        def is_small_enough(x):
            """Returns True if individuals are <= upper bounds."""
            return np.all(
                [np.less_equal(x[:, key], self.upper[key])
                 for key in self.upper.keys()],
                axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]),
            axis=0
        )


class Leaf(Node):
    """Class representing a leaf in a decision tree."""

    def __init__(self, value, depth=None):
        """Initializes the Leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """String representation of the leaf."""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """Returns the leaf itself in a list."""
        return [self]

    def update_bounds_below(self):
        """Base case for recursive bounds update."""
        pass


class Decision_Tree():
    """Class representing a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes the Decision Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion

    def __str__(self):
        """String representation of the tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Returns the list of all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Starts the recursive bounds update from the root."""
        self.root.update_bounds_below()

    def update_indicators(self):
        """Updates indicator functions for all leaves."""
        self.update_bounds()
        for leaf in self.get_leaves():
            leaf.update_indicator()

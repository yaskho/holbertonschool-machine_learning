#!/usr/bin/env python3
"""
Module to build a Decision Tree with leaf traversal capabilities
"""
import numpy as np


class Node:
    """Class representing an internal node in a decision tree"""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initializes the Node"""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def left_child_add_prefix(self, text):
        """Adds prefix for left child visualization"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("    |  " + x) + "\n"
        return (new_text)

    def right_child_add_prefix(self, text):
        """Adds prefix for right child visualization"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("       " + x) + "\n"
        return (new_text)

    def __str__(self):
        """String representation of the node"""
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
        """Returns the list of all leaves below this node"""
        leaves = []
        if self.left_child:
            leaves += self.left_child.get_leaves_below()
        if self.right_child:
            leaves += self.right_child.get_leaves_below()
        return leaves


class Leaf(Node):
    """Class representing a leaf in a decision tree"""

    def __init__(self, value, depth=None):
        """Initializes the Leaf"""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def __str__(self):
        """String representation of the leaf"""
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        """Returns the leaf itself in a list"""
        return [self]


class Decision_Tree():
    """Class representing a decision tree"""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initializes the Decision Tree"""
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
        self.predict = None

    def __str__(self):
        """String representation of the tree"""
        return self.root.__str__()

    def get_leaves(self):
        """Returns the list of all leaves in the tree"""
        return self.root.get_leaves_below()

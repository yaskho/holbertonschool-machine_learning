#!/usr/bin/env python3
"""
Decision tree implementation with printable structure.
"""

import numpy as np


class Node:
    """
    Represents an internal node in a decision tree.
    """

    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        is_root=False,
        depth=0
    ):
        """
        Initialize a Node.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Compute the maximum depth reachable below this node.
        """
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node.
        """
        count = (
            self.left_child.count_nodes_below(only_leaves=only_leaves)
            + self.right_child.count_nodes_below(only_leaves=only_leaves)
        )
        if not only_leaves:
            count += 1
        return count

    def left_child_add_prefix(self, text):
        """
        Add prefix formatting for the left child.
        """
        lines = text.split("\n")
        new_text = "    +--->" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "    |      " + line + "\n"
        return new_text.rstrip("\n")

    def right_child_add_prefix(self, text):
        """
        Add prefix formatting for the right child.
        """
        lines = text.split("\n")
        new_text = "    +--->" + lines[0] + "\n"
        for line in lines[1:]:
            new_text += "           " + line + "\n"
        return new_text.rstrip("\n")

    def __str__(self):
        """
        Return a string representation of the subtree rooted at this node.
        """
        if self.is_root:
            text = f"root [feature={self.feature}, threshold={self.threshold}]"
        else:
            text = f"-> node [feature={self.feature}, threshold={self.threshold}]"

        left_str = self.left_child.__str__()
        right_str = self.right_child.__str__()

        text += "\n" + self.left_child_add_prefix(left_str)
        text += "\n" + self.right_child_add_prefix(right_str)

        return text


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count this leaf.
        """
        return 1

    def __str__(self):
        """
        Return string representation of a leaf.
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    Decision Tree model.
    """

    def __init__(
        self,
        max_depth=10,
        min_pop=1,
        seed=0,
        split_criterion="random",
        root=None
    ):
        """
        Initialize the Decision Tree.
        """
        self.rng = np.random.default_rng(seed)
        self.root = root if root else Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
        Return the depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """
        Return string representation of the decision tree.
        """
        return self.root.__str__()

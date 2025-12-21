#!/usr/bin/env python3
"""
Decision tree implementation with node and leaf counting utilities.
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

        Args:
            feature: Feature index used for splitting.
            threshold: Threshold value for the split.
            left_child: Left child node.
            right_child: Right child node.
            is_root (bool): Indicates whether this node is the root.
            depth (int): Depth of the node in the tree.
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

        Returns:
            int: Maximum depth of the subtree rooted at this node.
        """
        return max(
            self.left_child.max_depth_below(),
            self.right_child.max_depth_below()
        )

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves below this node.
        """
        count = (
            self.left_child.count_nodes_below(only_leaves=only_leaves)
            + self.right_child.count_nodes_below(only_leaves=only_leaves)
        )

        if not only_leaves:
            count += 1

        return count


class Leaf(Node):
    """
    Represents a leaf node in a decision tree.
    """

    def __init__(self, value, depth=None):
        """
        Initialize a Leaf.

        Args:
            value: Predicted value at the leaf.
            depth (int): Depth of the leaf in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Return the depth of the leaf.

        Returns:
            int: Depth of the leaf.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """
        Count this leaf node.

        Args:
            only_leaves (bool): Included for API consistency.

        Returns:
            int: Always 1 for a leaf.
        """
        return 1


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

        Args:
            max_depth (int): Maximum depth of the tree.
            min_pop (int): Minimum population to allow a split.
            seed (int): Random seed.
            split_criterion (str): Criterion used for splitting.
            root (Node): Root node of the tree.
        """
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

    def depth(self):
        """
        Compute the depth of the decision tree.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """
        Count the number of nodes in the decision tree.

        Args:
            only_leaves (bool): If True, count only leaf nodes.

        Returns:
            int: Number of nodes or leaves in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

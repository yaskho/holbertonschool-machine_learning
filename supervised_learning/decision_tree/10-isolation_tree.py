#!/usr/bin/env python3
"""
Module to implement an Isolation Random Tree for outlier detection.
"""
import numpy as np
Node = __import__('8-build_decision_tree').Node
Leaf = __import__('8-build_decision_tree').Leaf


class Isolation_Random_Tree():
    """Class representing an isolation random tree."""

    def __init__(self, max_depth=10, seed=0, root=None):
        """Initializes the Isolation Random Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.max_depth = max_depth
        self.predict = None
        self.min_pop = 1

    def __str__(self):
        """String representation of the tree (same as Decision_Tree)."""
        return self.root.__str__()

    def depth(self):
        """Returns the maximum depth of the tree."""
        return max(leaf.depth for leaf in self.get_leaves())

    def count_nodes(self, only_leaves=False):
        """Counts total nodes or leaves (same as Decision_Tree)."""
        if only_leaves:
            return len(self.get_leaves())

        def _count(node):
            if node.is_leaf:
                return 1
            return 1 + _count(node.left_child) + _count(node.right_child)
        return _count(self.root)

    def update_bounds(self):
        """Updates bounds for all nodes (same as Decision_Tree)."""
        self.root.update_bounds_below()

    def get_leaves(self):
        """Returns list of all leaves (same as Decision_Tree)."""
        return self.root.get_leaves_below()

    def update_predict(self):
        """Updates vectorized prediction function (same as Decision_Tree)."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            indicators = np.array([leaf.indicator(A) for leaf in leaves])
            values = np.array([leaf.value for leaf in leaves])
            return np.dot(values, indicators)

        self.predict = predict_func

    def np_extrema(self, arr):
        """Returns min and max of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Random split criterion (same as Decision_Tree)."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population]
            )
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child where the value is the depth."""
        # For Isolation Trees, the value stored is the depth of the leaf
        leaf_child = Leaf(value=node.depth + 1)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node (same as Decision_Tree)."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit_node(self, node):
        """Recursively fits the isolation tree."""
        node.feature, node.threshold = self.random_split_criterion(node)

        feat_vals = self.explanatory[:, node.feature]
        left_pop = np.logical_and(node.sub_population,
                                  feat_vals > node.threshold)
        right_pop = np.logical_and(node.sub_population,
                                   feat_vals <= node.threshold)

        # In isolation trees, we stop if depth limit reached or population is 1
        def check_is_leaf(pop, depth):
            pop_size = np.sum(pop)
            if pop_size <= 1 or depth >= self.max_depth:
                return True
            return False

        # Is left node a leaf?
        if check_is_leaf(left_pop, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        # Is right node a leaf?
        if check_is_leaf(right_pop, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def fit(self, explanatory, verbose=0):
        """Trains the isolation tree on the explanatory data."""
        self.explanatory = explanatory
        # Initialize sub_population for root
        self.root.sub_population = np.ones(explanatory.shape[0], dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")

#!/usr/bin/env python3
"""
Module to build a Decision Tree with Gini split criterion optimization.
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
            """Check if features are greater than lower bounds."""
            return np.all(
                [np.greater(x[:, key], self.lower[key])
                 for key in self.lower.keys()],
                axis=0
            )

        def is_small_enough(x):
            """Check if features are less than or equal to upper bounds."""
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
        self.predict = None

    def depth(self):
        """Returns the maximum depth of the tree."""
        return max(leaf.depth for leaf in self.get_leaves())

    def count_nodes(self, only_leaves=False):
        """Counts total nodes or leaves in the tree."""
        if only_leaves:
            return len(self.get_leaves())

        def _count(node):
            if node.is_leaf:
                return 1
            return 1 + _count(node.left_child) + _count(node.right_child)
        return _count(self.root)

    def __str__(self):
        """String representation of the tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Returns the list of all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Starts the recursive bounds update from the root."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Updates the vectorized prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        def predict_func(A):
            indicators = np.array([leaf.indicator(A) for leaf in leaves])
            values = np.array([leaf.value for leaf in leaves])
            return np.dot(values, indicators)

        self.predict = predict_func

    def fit(self, explanatory, target, verbose=0):
        """Trains the decision tree on a dataset."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def np_extrema(self, arr):
        """Returns min and max of an array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Randomly selects a feature and threshold for splitting."""
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

    def possible_thresholds(self, node, feature):
        """Calculates midpoints between unique feature values."""
        values = np.unique((self.explanatory[:, feature])[node.sub_population])
        return (values[1:] + values[:-1]) / 2

    def Gini_split_criterion_one_feature(self, node, feature):
        """Calculates the best Gini threshold for a single feature."""
        sub_explanatory = self.explanatory[:, feature][node.sub_population]
        sub_target = self.target[node.sub_population]
        thresholds = self.possible_thresholds(node, feature)

        if thresholds.size == 0:
            return 0, np.inf

        classes = np.unique(sub_target)

        # 3D broadcasting
        cl_ind = sub_target[:, None, None] == classes[None, None, :]
        th_ind = sub_explanatory[:, None, None] > thresholds[None, :, None]

        Left_F = np.logical_and(cl_ind, th_ind)
        Right_F = np.logical_and(cl_ind, ~th_ind)

        left_counts = np.sum(Left_F, axis=0)
        right_counts = np.sum(Right_F, axis=0)

        n_left = np.sum(left_counts, axis=1)
        n_right = np.sum(right_counts, axis=1)
        n_total = n_left + n_right

        with np.errstate(divide='ignore', invalid='ignore'):
            gini_l = 1 - np.sum((left_counts / n_left[:, None])**2, axis=1)
            gini_r = 1 - np.sum((right_counts / n_right[:, None])**2, axis=1)

        gini_avg = (n_left * gini_l + n_right * gini_r) / n_total

        best_idx = np.argmin(gini_avg)
        return thresholds[best_idx], gini_avg[best_idx]

    def Gini_split_criterion(self, node):
        """Finds the overall best feature and threshold using Gini."""
        X = np.array([self.Gini_split_criterion_one_feature(node, i)
                      for i in range(self.explanatory.shape[1])])
        i = np.argmin(X[:, 1])
        return i, X[i, 0]

    def fit_node(self, node):
        """Recursively fits a node using the chosen split criterion."""
        node.feature, node.threshold = self.split_criterion(node)

        feat_vals = self.explanatory[:, node.feature]
        left_pop = np.logical_and(node.sub_population,
                                  feat_vals > node.threshold)
        right_pop = np.logical_and(node.sub_population,
                                   feat_vals <= node.threshold)

        def check_is_leaf(pop, depth):
            pop_size = np.sum(pop)
            if pop_size == 0 or pop_size < self.min_pop:
                return True
            if depth >= self.max_depth:
                return True
            if np.unique(self.target[pop]).size == 1:
                return True
            return False

        if check_is_leaf(left_pop, node.depth + 1):
            node.left_child = self.get_leaf_child(node, left_pop)
        else:
            node.left_child = self.get_node_child(node, left_pop)
            self.fit_node(node.left_child)

        if check_is_leaf(right_pop, node.depth + 1):
            node.right_child = self.get_leaf_child(node, right_pop)
        else:
            node.right_child = self.get_node_child(node, right_pop)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Creates a leaf child with the most frequent class value."""
        target_subset = self.target[sub_population]
        if target_subset.size == 0:
            value = 0
        else:
            value = np.bincount(target_subset).argmax()

        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Creates an internal node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        """Calculates prediction accuracy."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

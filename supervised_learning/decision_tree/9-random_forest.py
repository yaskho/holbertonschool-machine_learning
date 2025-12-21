#!/usr/bin/env python3
"""
Module to implement a Random Forest classifier.
"""
import numpy as np
Decision_Tree = __import__('8-build_decision_tree').Decision_Tree


class Random_Forest():
    """Class representing a random forest of decision trees."""

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """Initializes the Random Forest."""
        self.numpy_predicts = []
        self.target = None
        self.numpy_preds = None
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.seed = seed

    def predict(self, explanatory):
        """
        Predicts classes for the explanatory data by voting.
        """
        # Generate predictions for each tree in the forest.
        # all_preds shape: (n_trees, n_samples)
        all_preds = np.array([p(explanatory) for p in self.numpy_preds])

        # Calculate the mode (most frequent) prediction for each example.
        def get_mode(column):
            """Returns the most frequent value in a 1D array."""
            return np.bincount(column.astype(int)).argmax()

        # Apply mode calculation along the tree axis (axis 0)
        return np.apply_along_axis(get_mode, axis=0, arr=all_preds)

    def fit(self, explanatory, target, n_trees=100, verbose=0):
        """Trains the random forest."""
        self.target = target
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []
        accuracies = []

        for i in range(n_trees):
            T = Decision_Tree(max_depth=self.max_depth,
                              min_pop=self.min_pop,
                              seed=self.seed + i)
            T.fit(explanatory, target)
            self.numpy_preds.append(T.predict)
            depths.append(T.depth())
            nodes.append(T.count_nodes())
            leaves.append(T.count_nodes(only_leaves=True))
            accuracies.append(T.accuracy(T.explanatory, T.target))

        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Mean depth                     : "
                  f"{np.array(depths).mean()}")
            print(f"    - Mean number of nodes           : "
                  f"{np.array(nodes).mean()}")
            print(f"    - Mean number of leaves          : "
                  f"{np.array(leaves).mean()}")
            print(f"    - Mean accuracy on training data : "
                  f"{np.array(accuracies).mean()}")
            print(f"    - Accuracy of the forest on td   : "
                  f"{self.accuracy(self.explanatory, self.target)}")

    def accuracy(self, test_explanatory, test_target):
        """Calculates the accuracy of the forest predictions."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size

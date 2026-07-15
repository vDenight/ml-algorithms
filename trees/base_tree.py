import numpy as np
import numpy.typing as npt


def calculate_node_gini(labels: npt.NDArray[np.int64]) -> np.float64:
    """
    Calculates the Gini Impurity of a node.

    The formula is: 1 - SUM(from i to C)(p_i^2) where:
    C is the number of classes,
    p_i is the probability of each class,
    p_i = number of samples belonging to the i-class divided by the number of all samples of the node.
    """
    num_samples = labels.shape[0]

    if num_samples == 0:
        return np.float64(0.0)

    _, counts = np.unique(labels, return_counts=True)

    probabilities = counts / num_samples
    return 1.0 - np.sum(probabilities ** 2)


def calculate_weighted_gini(feature: np.uint64, threshold: np.float64, X: npt.NDArray[np.float64],
                            labels: npt.NDArray[np.uint64], min_samples_leaf: int) -> np.float64:
    """
    Calculates the weighted Gini Impurity of a node.

    The formula is: N_left/N * GI_left + N_right/N * GI_right where:
    N_{side} is the number of samples belonging to a specific side,
    GI_{side} is the Gini Impurity of each side child node,
    N is the number of all samples of the node,
    """
    n_total = labels.shape[0]

    if n_total == 0:
        return np.float64(0.0)

    left_mask = X[:, feature] <= threshold
    left_labels = labels[left_mask]
    right_labels = labels[~left_mask]

    if left_labels.shape[0] < min_samples_leaf or right_labels.shape[0] < min_samples_leaf:
        return np.float64(np.inf)

    gini_left = calculate_node_gini(left_labels)
    gini_right = calculate_node_gini(right_labels)

    return (left_labels.shape[0] / n_total * gini_left) + (right_labels.shape[0] / n_total * gini_right)


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, certainty=None):
        self.feature = feature  # Index of the feature to split on
        self.threshold = threshold  # The value to split that feature at
        self.left = left  # The left child Node
        self.right = right  # the right child Node

        self.value = value  # the majority class (only populated if it is a leaf)
        self.certainty = certainty  # optional value to get a certainty like 75% of accurate guess

    def predict(self, x: npt.NDArray[np.float64]) -> int:
        """
        Predicts the label for a single 1D sample array.
        """
        if self.value is not None:
            return self.value

        if x[self.feature] <= self.threshold:
            return self.left.predict(x)

        return self.right.predict(x)

    def predict_with_certainty(self, x) -> tuple[int, float]:
        if self.value is not None:
            return self.value, self.certainty

        if x[self.feature] <= self.threshold:
            return self.left.predict_with_certainty(x)
        return self.right.predict_with_certainty(x)

    def is_leaf_node(self):
        """Helper method to check if this node is a leaf."""
        return self.value is not None


class BaseTree:
    """
    This is an implementation for a simple decision tree,
    which uses the gini impurity for choosing the right nodes.
    """

    def __init__(self, max_depth: int, min_samples_split_ratio: float, min_samples_leaf_ratio: float) -> None:
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than 0")
        if min_samples_split_ratio < 0 or min_samples_split_ratio >= 1:
            raise ValueError("min_samples_split_ratio must be between 0 and 1")
        if min_samples_leaf_ratio < 0 or min_samples_leaf_ratio >= 1:
            raise ValueError("min_samples_leaf_ratio must be between 0 and 1")

        self.max_depth = max_depth
        self.min_samples_split_ratio = min_samples_split_ratio
        self.min_samples_leaf_ratio = min_samples_leaf_ratio
        self.root = None

    def fit(self, X: npt.NDArray[np.float64], labels: npt.NDArray[np.uint64]):
        # create a root node and all nodes subsequently using a recursion
        min_samples_split = int(X.shape[0] * self.min_samples_split_ratio)
        min_samples_leaf = int(X.shape[0] * self.min_samples_leaf_ratio)
        self.root = self._build_tree(X, labels, min_samples_split, min_samples_leaf)

    def _build_tree(self, X: npt.NDArray[np.float64], labels: npt.NDArray[np.uint64], depth=0,
                    min_samples_split: int = 0, min_samples_leaf: int = 0) -> Node:
        """
        This is a recursive function that builds the entire tree, based on its current X, labels and depth.
        what it does:
        - first it checks depth to know if this must be a leaf node, if yes it returns a new node, that's leaf
        - if not, it checks for the best feature and threshold to split and creates a node based on that
        - then it checks for a split of the node
        """
        # case 1: we reached a max_depth and we must return a (forced) leaf node
        if depth >= self.max_depth:
            return self._create_leaf(labels)

        # case 2: this is a pure node, so we return a leaf (early stopping)
        if np.all(labels == labels[0]):
            return self._create_leaf(labels)

        # case 3: this node is not suitable for a split min_samples_split not satisfied
        if X.shape[0] < min_samples_split:
            return self._create_leaf(labels)

        # not a pure node so we look for a best split strategy
        node = self._get_best_candidate(X, labels)

        left_mask = X[:, node.feature] <= node.threshold

        # we additionally check if our threshold is splitting any data
        if np.any(left_mask) and np.any(~left_mask):
            # case 4: we get a split so we build additional nodes as left and right children
            node.left = self._build_tree(X[left_mask], labels[left_mask], depth + 1, min_samples_split, min_samples_leaf)
            node.right = self._build_tree(X[~left_mask], labels[~left_mask], depth + 1, min_samples_split, min_samples_leaf)
        else:
            # case 5: best feature/threshold combo isn't able to differentiate any two sub-groups
            # in this case we need to (forcefully) transform it node as further splits don't make any sense
            return self._create_leaf(labels)

        # after completing all recursion calls we can return the processed node
        return node

    def _create_leaf(self, labels: npt.NDArray[np.uint64]) -> Node:
        """Helper to calculate the most frequent label and create a leaf node."""
        values, counts = np.unique(labels, return_counts=True)
        max_index = np.argmax(counts)

        most_frequent = values[max_index]
        certainty = counts[max_index] / labels.shape[0]

        return Node(value=most_frequent, certainty=certainty)

    def _get_best_candidate(self, X: npt.NDArray[np.float64], labels: npt.NDArray[np.uint64], min_samples_leaf) -> Node:
        """Helper that tries to find the best node to split on"""
        best_candidate = {"threshold": np.float64(np.inf), "feature": np.uint64(0), "weighted_gini": np.float64(np.inf)}
        for i in range(X.shape[1]):
            unique_col_vals_sorted = np.unique(X[:, i])

            # explicitly skip this feature if all the label values are identical
            if len(unique_col_vals_sorted) <= 1:
                continue

            thresholds: npt.NDArray[np.float64] = (unique_col_vals_sorted[:-1] + unique_col_vals_sorted[1:]) / 2
            for threshold in thresholds:
                weighted_gini = calculate_weighted_gini(i, threshold, X, labels, min_samples_leaf)
                if weighted_gini < best_candidate["weighted_gini"]:  # comparing to the best score so far
                    best_candidate = {"threshold": threshold, "feature": i, "weighted_gini": weighted_gini}

        return Node(feature=best_candidate["feature"], threshold=best_candidate["threshold"])

    def print_tree(self, node=None, depth=0):
        """
        Prints the tree structure recursively to the console.
        Call this without arguments: my_tree.print_tree()
        """
        # If it's the first call, start at the root
        if node is None:
            if self.root is None:
                print("The tree is empty. Please call fit() first.")
                return
            node = self.root

        # Create the visual indentation based on current depth
        indent = "|   " * depth

        # BASE CASE: If it's a leaf, print the prediction
        if node.is_leaf_node():
            print(f"{indent}-> Leaf: Predict Class {node.value} (Certainty: {node.certainty:.2%})")
            return

        # RECURSIVE STEP: Print the split condition, then traverse left and right
        print(f"{indent}[Depth {depth}] If Feature {node.feature} <= {node.threshold:.4f}:")
        self.print_tree(node.left, depth + 1)

        print(f"{indent}[Depth {depth}] Else (Feature {node.feature} > {node.threshold:.4f}):")
        self.print_tree(node.right, depth + 1)

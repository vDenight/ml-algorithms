import numpy as np
import numpy.typing as npt

from trees.base_tree import BaseTree, Node

class RandomForestTree(BaseTree):
    def __init__(self, max_depth, max_features, random_state, *, min_samples_split=0, min_samples_leaf=0,
                 min_samples_split_ratio: float | None = None,
                 min_samples_leaf_ratio: float | None = None) -> None:
        self.max_features = max_features
        self.random_state = random_state
        super().__init__(max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_samples_split_ratio=min_samples_split_ratio, min_samples_leaf_ratio=min_samples_leaf_ratio)

    def _get_best_candidate(self, X: npt.NDArray[np.float64], labels: npt.NDArray[np.uint64], min_samples_leaf) -> Node:
        """we add a wrapper to choose only from selected features"""
        feats_selected = np.random.permutation(X.shape[1])[:self.max_features]
        X_selection = X[:, feats_selected]

        node = super()._get_best_candidate(X_selection, labels, min_samples_leaf)

        if node.feature is not None: # we need to map the node back to the original feature index
            node.feature = feats_selected[node.feature]

        return node

class RandomForest:
    def __init__(self, *, n_estimators, max_features, max_depth=10, min_samples_split=0, min_samples_leaf=0, random_state=42):
        self.n_estimators = n_estimators
        self.estimators: list[RandomForestTree] = []
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.n_labels = 0

    def fit(self, X: npt.NDArray[np.float64], labels: npt.NDArray[np.uint64], *, min_samples_split=0, min_samples_leaf=0):
        self.n_labels = int(np.max(labels)) + 1

        for _ in range(self.n_estimators):
            bs_set, bs_labels = self._bootstrap_set(X, labels)
            new_tree = RandomForestTree(self.max_depth, self.max_features, self.random_state, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
            new_tree.fit(bs_set, bs_labels)
            self.estimators.append(new_tree)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.uint64]:
        """predict entire matrix"""
        return np.array([self.predict_single(x) for x in X])

    def predict_proba(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """predict entire matrix (probabilities)"""
        return np.array([self.predict_proba_single(x) for x in X])

    def predict_single(self, x: npt.NDArray[np.float64]) -> np.uint64:
        """we are using soft voting with predict_proba"""
        all_probs = np.zeros(self.n_labels)
        for tree in self.estimators:
            if tree.root is None:
                raise RuntimeError("The labels cannot be predicted, because one of the trees is malformed")

            probs = tree.root.predict_proba(x)

            if probs is None:
                raise RuntimeError("The labels cannot be predicted, because one of the trees is malformed")

            all_probs += probs

        return np.uint64(np.argmax(all_probs))

    def predict_proba_single(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """we are using soft voting with predict_proba"""
        all_probs = np.zeros(self.n_labels)
        for tree in self.estimators:
            if tree.root is None:
                raise RuntimeError("The labels cannot be predicted, because one of the trees is malformed")

            probs = tree.root.predict_proba(x)

            if probs is None:
                raise RuntimeError("The labels cannot be predicted, because one of the trees is malformed")

            all_probs += probs

        return all_probs / np.sum(all_probs)

    @staticmethod
    def _bootstrap_set(X: npt.NDArray[np.float64], labels: npt.NDArray[np.uint64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint64]]:
        chosen_indexes = np.random.randint(0, X.shape[0], size=X.shape[0])
        return X[chosen_indexes], labels[chosen_indexes]

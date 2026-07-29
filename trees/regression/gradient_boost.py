import numpy as np

from trees.regression.base_regression_tree import RegressionTree
from beartype import beartype
from jaxtyping import Float

class GradientBoost:
    def __init__(
        self,
        n_estimators: int,
        learning_rate: float,
        max_depth: int,
        min_samples_split: int,
        min_samples_leaf: int,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate: float = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.F_0: float = 0.
        self.estimators: list[RegressionTree] = []

    @beartype
    def predict(self, X: Float[np.ndarray, "N F"]) -> Float[np.ndarray, " N"]:
        pred = np.full(X.shape[0], self.F_0)
        for tree in self.estimators:
            pred += self.learning_rate * tree.predict(X)

        return pred

    @beartype
    def fit(self, X: Float[np.ndarray, "N F"], y: Float[np.ndarray, " N"]):

        self.F_0 = np.mean(y)
        residuals = y - self.F_0

        for i in range(self.n_estimators):
            self.estimators.append(self.build_estimator(X, residuals))
            residuals -= self.learning_rate * self.estimators[-1].predict(X)


    @beartype
    def build_estimator(self, X: Float[np.ndarray, "N F"], residuals: Float[np.ndarray, " N"]):
        new_tree = RegressionTree(self.max_depth, self.min_samples_split, self.min_samples_leaf)
        new_tree.fit(X, residuals)
        return new_tree

from beartype import beartype
import numpy as np
from jaxtyping import Float
from abc import ABC, abstractmethod

class Node(ABC):
    pass

    @abstractmethod
    def predict(self, x: Float[np.ndarray, " K"]) -> float:
        pass

class LeafNode(Node):
    def __init__(self, mean_value: float):
        self.mean_value = mean_value

    @beartype
    def predict(self, x: Float[np.ndarray, " K"]) -> float:
        return self.mean_value

class DecisionNode(Node):
    def __init__(self, threshold: float, feature: int, left_child: Node | None = None, right_child: Node | None = None):
        self.threshold = threshold
        self.feature = feature
        self.left_child = left_child
        self.right_child = right_child

    @beartype
    def predict(self, x: Float[np.ndarray, " K"]) -> float:
        if self.left_child is None or self.right_child is None:
            raise RuntimeError("DecisionNode has no child nodes initialized")

        feat_val: float = x[self.feature]
        if feat_val < self.threshold:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

class RegressionTree:
    def __init__(self, max_depth: int, min_samples_split: int, min_samples_leaf: int):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.root = None

    def find_best_candidate_naive(self, X: Float[np.ndarray, "N F"], y: Float[np.ndarray, " N"]):
        best_candidate = {
            "threshold": float("inf"),
            "feature": -1,
            "MSE": float("inf"),
        }

        for col in range(X.shape[1]):
            unique_col_vals_sorted = np.unique(X[:, col]) # np unique sorts those as well

            if len(unique_col_vals_sorted) <= 1:
                continue

            thresholds = (unique_col_vals_sorted[:-1] + unique_col_vals_sorted[1:]) / 2
            for threshold in thresholds:
                split_MSE = self.calculate_split_MSE(X[:, col], y, threshold)
                if split_MSE < best_candidate["MSE"]:
                    best_candidate["threshold"] = threshold
                    best_candidate["feature"] = col
                    best_candidate["MSE"] = split_MSE

        return best_candidate

    @staticmethod
    def calculate_split_MSE(x: Float[np.ndarray, " N"], y: Float[np.ndarray, " N"], threshold: float) -> float:
        left_mask = x < threshold
        left_y = y[left_mask]
        left_avg = np.mean(left_y)
        right_y = y[~left_mask]
        right_avg = np.mean(right_y)

        return (np.sum(np.square(left_y - left_avg)) + np.sum(np.square(right_y - right_avg))) / y.shape[0]

    def find_best_candidate_optimized(self, X: Float[np.ndarray, "N F"], y: Float[np.ndarray, " N"], min_samples_leaf: int=1):
        """In most videos explaining the Regression Tree the algorithm
        for finding the best candidate for a sleep is straightforward,
        - just calculate all values and compare
        There is however a cool math trick that allows us to speed up the calculation
        for each threshold massively.
        It all comes down to the MSE formula:
        MSE(side) = SUM((y_i-y_avg)^2), after some algebraic manipulation,
        we can arrive at the following:
        SUM((y_i-y_avg)^2) = SUM((y_i)^2) - ((SUM(y_i))^2)/n,
        this means that to calculate the MSE value for each threshold we just need
        to keep track of the following:
        n -> number of elements in a split
        SUM(y_i^2) -> sum of the squared elements
        SUM(y_i) -> sum of the elements
        We don't really need to calculate the mean value for each split, and by moving a threshold,
        we can arrive at new MSE value by simple arithmetic (ex. add y_i to one side subtract from the other)
        Additionally, we can use np.cumsum (cumulative sum) to quickly calculate all the MSE values"""

        best_candidate = {
            "threshold": float("inf"),
            "feature": -1,
            "MSE": float("inf"),
        }

        n_samples, n_features = X.shape

        for col in range(n_features):
            x = X[:, col]

            n = np.arange(min_samples_leaf, n_samples - min_samples_leaf + 1)

            sort_indices = np.argsort(x)
            x_sorted = x[sort_indices]
            y_sort = y[sort_indices]

            left_sums_y = np.cumsum(y_sort)
            left_sums_sq_y = np.cumsum(np.square(y_sort))

            # extract the sum of all before cutting
            y_i_sum = left_sums_y[-1]
            y_i_sq_sum = left_sums_sq_y[-1]

            start_idx = min_samples_leaf - 1
            end_idx = n_samples - min_samples_leaf

            if start_idx >= end_idx:
                continue

            # now we cut elements accordingly to min_samples_leaf parameter
            # to exclude the splits witch don't conform to it
            left_sums_y = left_sums_y[start_idx:end_idx]
            left_sums_sq_y = left_sums_sq_y[start_idx:end_idx]

            right_sums_y = y_i_sum - left_sums_y
            right_sums_sq_y = y_i_sq_sum -left_sums_sq_y

            mse_vec = left_sums_sq_y - (left_sums_y ** 2) / n + right_sums_sq_y - (right_sums_y ** 2) / (x.shape[0] - n)

            # we additionally perform some boolean masking to exclude invalid (same value) splits
            invalid_splits = x_sorted[start_idx:end_idx] == x_sorted[start_idx + 1:end_idx + 1]
            mse_vec[invalid_splits] = np.inf

            if np.all(np.isinf(mse_vec)):
                continue

            best_mse_index = np.argmin(mse_vec)
            mse = mse_vec[best_mse_index]
            if mse < best_candidate["MSE"]:
                best_candidate["MSE"] = mse
                best_candidate["feature"] = col
                best_candidate["threshold"] = (x_sorted[start_idx + best_mse_index] + x_sorted[start_idx + best_mse_index + 1]) / 2

        return best_candidate
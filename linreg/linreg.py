from typing import Generator

import numpy as np
import numpy.typing as ntp

seed_value = 42
rng = np.random.default_rng(seed=seed_value)


class LinearRegression:
    def __init__(self, feat_len: int, l1_coeff: float = 0.0, l2_coeff: float = 0.0):
        self.w_len = feat_len + 1
        self.weights = (
            rng.random(size=(1, self.w_len)) * 2
        ) - 1  # weights coded as row vector
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff

    def fit_generator(
        self,
        X: ntp.NDArray[np.float64],
        y: ntp.NDArray[np.float64],
        learning_rate: float,
        batch_size: int,
        epochs: int,
    ) -> Generator[ntp.NDArray[np.float64]]:
        """
        This method is used as a generator to fit our model to the training data.
        We are using a generator to be able to get more insight into how
        the model behaves during training.
        """
        yield self.weights
        n_samples = X.shape[0]
        for _epoch in range(epochs):
            iterations = n_samples // batch_size
            for iteration in range(iterations):
                start_index = iteration * batch_size
                end_index = (
                    start_index + batch_size
                    if start_index + batch_size < n_samples
                    else n_samples
                )

                X_partial = X[start_index:end_index]
                y_partial = y[start_index:end_index]

                grad = self.mse_gradient(X_partial, y_partial)

                self.weights = (
                    self.weights + (-learning_rate) * grad
                )  # updating the weights.
                yield self.weights

    def predict(self, X: ntp.NDArray[np.float64]) -> ntp.NDArray[np.float64]:
        """
        This method predicts the output of the linear regression model.
        The shape of the result is (n_samples, 1) vector.
        """
        X_w_ones = np.c_[
            X, np.ones(X.shape[0])
        ]  # so we make this magic to add a bias column of ones
        return X_w_ones @ self.weights.T  # predictions come out as column vectors

    def calculate_mse(
        self, X: ntp.NDArray[np.float64], y_actual: ntp.NDArray[np.float64]
    ) -> np.float64:
        """
        So we are using MSE as our cost function.
        It's nice to have a way to look at its current state.
        """
        if X.shape[0] != y_actual.shape[0]:
            raise ValueError("X and y_actual must have same number of columns")

        y_predict = self.predict(X)

        return np.float64(np.mean((y_actual - y_predict) ** 2))

    def cost_function(
        self, X: ntp.NDArray[np.float64], y_actual: ntp.NDArray[np.float64]
    ) -> np.float64:
        """
        This is MSE with addition of the regularization.
        """
        cost = self.calculate_mse(X, y_actual)
        cost += self.l1_coeff * np.sum(np.absolute(self.weights))
        cost += self.l2_coeff * np.sum(np.square(self.weights))
        return cost

    def mse_gradient(
        self, X: ntp.NDArray[np.float64], y_actual: ntp.NDArray[np.float64]
    ) -> ntp.NDArray[np.float64]:
        """
        This method calculates the gradient of MSE cost function against
        the passed training data X and the actual outputs y.
        The result is (1, w_len) vector, same as the weight vector.
        """
        n_samples = X.shape[0]

        y_pred = self.predict(X)
        y_diff = (y_pred - y_actual).reshape(1, n_samples)  # reshape into row vector
        X_w_ones = np.c_[X, np.ones(X.shape[0])]

        grad = (2 / n_samples) * y_diff @ X_w_ones

        reg_vec = (
            self.l1_coeff * np.sign(self.weights) + 2 * self.l2_coeff * self.weights
        )

        # zero out the bias term
        reg_vec[0, -1] = 0.0

        return grad + reg_vec

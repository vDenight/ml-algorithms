from linreg.linreg import LinearRegression
import numpy as np
import numpy.typing as npt



class LogisticRegression:

    def __init__(self, n_features: int, threshold: np.float64 = 0.5, seed: int | None = None):
        self.linreg = LinearRegression(n_features, seed=seed)
        self.threshold = threshold

        self.rng = np.random.default_rng(seed=seed)

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], learning_rate: float,  batch_size: int, epochs: int):
        """ function used to fit the logistic regression model"""
        n_samples = X.shape[0]
        for _epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            iterations = int(np.ceil(n_samples / batch_size))

            for iteration in range(iterations):
                start_index = iteration * batch_size
                end_index = min(start_index + batch_size, n_samples)

                X_partial = X_shuffled[start_index:end_index]
                y_partial = y_shuffled[start_index:end_index]

                grad = self.cross_entropy_grad(X_partial, y_partial)

                self.linreg.weights = (
                        self.linreg.weights - (learning_rate * grad)
                )

    def predict_proba(self, X: npt.NDArray[np.float64]):
        """
        predicts the probability of a positive class
        """
        # first do the regression calculation
        y_unbound = self.linreg.predict(X)
        y_mapped = self._sigmoid(y_unbound)

        return y_mapped

    def predict(self, X: npt.NDArray[np.float64]):
        """
        predicts the class label
        """
        y_mapped = self.predict_proba(X)
        y_classified = np.where(y_mapped >= self.threshold, 1, 0)

        return y_classified

    def cross_entropy_grad(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]):
        """
        calculate the gradient of the cross-entropy loss
        """
        n_samples = X.shape[0]
        X_w_ones = np.c_[X, np.ones(X.shape[0])]
        y_diff = self.predict_proba(X) - y

        return (1/n_samples) * y_diff.T @ X_w_ones

    @staticmethod
    def _sigmoid(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return 1 / ( 1 + np.exp(-x))


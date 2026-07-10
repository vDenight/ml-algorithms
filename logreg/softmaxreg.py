import numpy as np
import numpy.typing as npt


class SoftMaxRegression:
    """
    this is a class for softmax regression.
    it's an extension of logistic regression but for multiple labels
    """
    def __init__(self, n_features: int, n_labels: int, seed: int | None = None):
        self.rng = np.random.default_rng(seed=seed)
        self.W = self.rng.random(size=(n_features + 1, n_labels))
        # so W is technically now a (n_feat + 1, n_label) matrix
        # so each column corresponds to the weights of different features of the single label node
        # and each row has the weights for the single feature but different label node

    def fit(self, X: npt.NDArray[np.float64], Y_actual: npt.NDArray[np.float64], learning_rate: float, batch_size: int,
            epochs: int):
        n_samples = X.shape[0]
        for _epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y_actual[indices]
            iterations = int(np.ceil(n_samples / batch_size))

            for iteration in range(iterations):
                start_index = iteration * batch_size
                end_index = min(start_index + batch_size, n_samples)

                X_partial = X_shuffled[start_index:end_index]
                Y_partial = Y_shuffled[start_index:end_index]

                grad = self._ce_gradient(X_partial, Y_partial)

                self.W = (
                        self.W - (learning_rate * grad)
                )

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        X_w_ones = np.c_[X, np.ones(X.shape[0])] # X_w_ones (h=n_samples, w=n_feat + 1)
        Z = X_w_ones @ self.W # Z (h=n_samples, w=n_labels)
        return self._softmax(Z)

    @staticmethod
    def _softmax(Z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        Z_shifted = Z - np.max(Z, axis=-1, keepdims=True) # subtracting max to avoid overflow
        # we keepdims=True to make sure it's a column vector with h=n_samples and w=1, so that
        # we can apply the same number for each row

        exp_Z = np.exp(Z_shifted)

        return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True) # Y_pred (h=n_samples, w=n_labels)
        # same logic regarding the keepdims

    def _ce_gradient(self, X: npt.NDArray[np.float64], Y_actual: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculating the gradient of the softmax loss function
        for this one we expect:
        X - input matrix (h=n_samples, w=n_features)
        Y - target output matrix (h=n_samples, w=n_labels)"""
        n_samples = X.shape[0]

        Y_diff = self.predict(X) - Y_actual # again Y_diff has size of Y, so (h=n_samples, w=n_labels)
        X_w_ones = np.c_[X, np.ones(n_samples)] # now it's (h=n_samples, w=n_feat+1)

        return (X_w_ones.T @ Y_diff) / n_samples
        # so we transpose X_w_ones to get (h=n_feat+1, w=n_samples) and multiply it with Y_diff (h=n_samples, w=n_labels).
        # Based on this we should get a Matrix of (h=n_feat+1, w=n_labels)
        # This size is (as expected) exactly as out Weight Matrix, for which we have calculated the gradient.


import numpy as np


class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]

        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column >= self.threshold] = -1

        return predictions


class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.trees = []
        self.alphas = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # 1. Initialize weights to 1/N
        weights = np.ones(n_samples) / n_samples

        for _ in range(self.n_estimators):
            stump = DecisionStump()
            min_error = float("inf")

            # --- FIND THE BEST STUMP ---
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)

                for threshold in thresholds:
                    p = 1
                    predictions = np.ones(n_samples)
                    predictions[X_column < threshold] = -1

                    # Calculate weighted error
                    error = np.sum(weights[y != predictions])

                    # Flip polarity if error is worse than random guessing
                    if error > 0.5:
                        error = 1.0 - error
                        p = -1

                    # Save the best split parameters
                    if error < min_error:
                        min_error = error
                        stump.polarity = p
                        stump.threshold = threshold
                        stump.feature_idx = feature_i

            # --- ADABOOST CORE MATH ---

            # 2. Calculate "Amount of Say" (alpha)
            # Add a tiny epsilon to prevent division by zero in perfect splits
            EPS = 1e-10
            alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))

            # Save the trained stump and its alpha
            self.trees.append(stump)
            self.alphas.append(alpha)

            # 3. Update sample weights
            stump_pred = stump.predict(X)
            # Incorrect predictions result in a positive exponent (weight increases)
            # Correct predictions result in a negative exponent (weight decreases)
            weights *= np.exp(-alpha * y * stump_pred)

            # 4. Normalize weights so they sum to 1
            weights /= np.sum(weights)

    def predict(self, X):
        # Fetch predictions from all trees: shape (n_trees, n_samples)
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        alphas_array = np.array(self.alphas)

        # Fast vectorized weighted sum: (n_samples,)
        weighted_sums = np.dot(tree_preds.T, alphas_array)

        # Return the sign (-1 or 1)
        return np.sign(weighted_sums)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    # Generate dummy data
    X, y = make_classification(n_samples=500, n_features=5, random_state=42)

    # Crucial step: map 0 to -1
    y = np.where(y == 0, -1, 1)

    # Train and predict
    clf = AdaBoost(n_estimators=15)
    clf.fit(X, y)
    predictions = clf.predict(X)

    print(f"Accuracy: {accuracy_score(y, predictions) * 100:.2f}%")
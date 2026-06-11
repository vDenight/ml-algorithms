import numpy as np


def main():
    print("Hello from ml-regression!")


if __name__ == "__main__":
    main()
    X = np.array([[1, 2, 3, 5],
                  [4, 5, 6, 5],
                  [7, 8, 9, 5]])
    y = np.array([1, 2, 3, 4])
    # width is 4, height is 3
    print(f"X - width is: {X.shape[1]}, height is: {X.shape[0]}")
    print(f"Y - shape is width is {y.shape[0]}")

    X = np.array([[2.5, 1.2],
                  [3.1, 4.5],
                  [0.8, 1.9]])

    # Create a 1D array of 1s matching the number of rows in X
    ones = np.ones(X.shape[0])

    # Append the ones to the END of the matrix (if bias is the last weight)
    X_with_bias = np.c_[X, ones]

    print(X_with_bias)


import numpy as np


def loss_(y, y_hat):
    """Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    The half mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """
    if y.size == 0 or y_hat.size == 0 or y.shape != y_hat.shape:
        return None
    return np.sum((y_hat - y) ** 2) / (2 * len(y))


if __name__ == "__main__":
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    # Example 1:
    print(loss_(X, Y))
    # # Output:
    # 2.142857142857143
    # Example 2:
    print(loss_(X, X))
    # # Output:
    # 0.0

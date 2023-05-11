import numpy as np
from tools import add_intercept


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """

    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None

    if x.ndim != 1 or x.size == 0 or theta.shape != (2, 1):
        return None

    X = np.column_stack((np.ones(len(x)), x))
    y_hat = np.dot(X, theta)

    return y_hat

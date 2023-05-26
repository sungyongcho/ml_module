import numpy as np
import math


def logistic_predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    m = len(x)  # Number of training examples

    # Add a column of ones to X as the first column
    x_prime = np.concatenate((np.ones((m, 1)), x), axis=1)

    return (1 / (1 + math.e ** (-np.dot(x_prime, theta))))

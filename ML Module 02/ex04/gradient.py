import numpy as np
from typing import Tuple


def gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.

    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.

    Return:
    The gradient as a numpy.array, a vector of dimensions n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible dimensions.
    None if x, y or theta is not of expected type.

    Raises:
    This function should not raise any Exception.
    """
    m = len(y)  # Number of training examples

    # Add a column of ones to X as the first column
    x_prime = np.concatenate((np.ones((m, 1)), x), axis=1)

    # Compute the difference between predicted and actual values
    diff = np.dot(x_prime, theta) - y

    # Compute the gradient
    gradient = (1/m) * np.dot(x_prime.T, diff)

    return gradient

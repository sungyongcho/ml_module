import numpy as np


def add_polynomial_features(x, power):
    """
    Add polynomial features to matrix x by raising its columns to
    every power in the range of 1 up to the power giveArgs:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    power: has to be an int, the power up to which the columns
    of matrix x are going to be raised.
    Returns:
    The matrix of polynomial features as a numpy.ndarray, of shape m * (np),
    containg the polynomial feature vaNone if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    # Check if x is an empty array
    if x.size == 0:
        return None
    if power == 0:
        return x

    polynomial_features = []
    for col in range(1, power + 1):
        polynomial_features.append(np.power(x, col))

    return np.hstack(polynomial_features)

import numpy as np


def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    # if not isinstance(x, np.array):
    # return None
    if not isinstance(x, np.ndarray):
        print("Invalid input: argument of ndarray type required")
        return None

    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    elif not x.ndim == 2:
        print("Invalid input: wrong shape of x", x.shape)
        return None
    intercept = np.ones((x.shape[0], 1))
    X = np.concatenate((intercept, x), axis=1)
    return X

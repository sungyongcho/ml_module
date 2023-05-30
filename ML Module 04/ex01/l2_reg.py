import numpy as np


def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if theta.size == 0:
        return None

    l2_reg = 0.0
    # for val in theta:
    #     print(val)
    #     l2_reg += val ** 2
    for val in theta[1:]:
        l2_reg += val ** 2

    return float(l2_reg)


def l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
    The L2 regularization as a float.
    None if theta in an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if theta.size == 0:
        return None

    # l2_reg = np.sum(np.power(theta, 2))
    l2_reg = np.sum(np.power(theta[1:], 2))

    return float(l2_reg)


if __name__ == "__main__":
    x = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # Example 1:
    print(iterative_l2(x))
    # # Output:
    # 911.0
    # Example 2:
    print(l2(x))
    # # Output:
    # 911.0
    y = np.array([3, 0.5, -6]).reshape((-1, 1))
    # Example 3:
    print(iterative_l2(y))
    # # Output:
    # 36.25
    # Example 4:
    print(l2(y))
    # # Output:
    # 36.25

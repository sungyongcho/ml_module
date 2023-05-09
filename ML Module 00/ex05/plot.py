import numpy as np
import matplotlib.pyplot as plt

from tools import add_intercept
from prediction import predict_


def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """

    # if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
    #     return None

    # if x.size == 0 or y.size == 0 or theta.size == 0:
    #     return None

    # if x.shape[0] != y.shape[0] or theta.shape != (2,):
    #     return None

    X = add_intercept(x)
    y_hat = np.dot(X, theta)

    plt.scatter(x, y)
    plt.plot(x, y_hat, color='orange')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.grid()
    plt.title('Linear Regression')
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    # Example 1:
    theta1 = np.array([[4.5], [-0.2]])
    plot(x, y, theta1)

    theta2 = np.array([[-1.5], [2]])
    plot(x, y, theta2)

    theta3 = np.array([[3], [0.3]])
    plot(x, y, theta3)

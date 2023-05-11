import numpy as np

import matplotlib.pyplot as plt
from vec_loss import loss_
from tools import add_intercept


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exception.
    """
    # Check if the inputs are numpy arrays and non-empty
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None

    if x.size == 0 or y.size == 0 or theta.size == 0:
        return None

    # Check if the shapes of x, y, and theta are correct
    if x.shape[0] != y.shape[0] or theta.shape != (2,):
        return None

    X = add_intercept(x)

    y_hat = np.dot(X, theta)
    # Wtf
    J = loss_(y, y_hat) * 2

    fig, ax = plt.subplots()

    ax.scatter(x, y)

    ax.plot(x, y_hat, color='orange')

    for i in range(len(x)):
        ax.plot([x[i], x[i]], [y[i], y_hat[i]], linestyle='--', color='red')

    plt.title("Cost: {:.6f}".format(J))
    plt.show()


if __name__ == "__main__":
    x = np.arange(1, 6)
    y = np.array([11.52434424, 10.62589482,
                 13.14755699, 18.60682298, 14.14329568])
    # Example 1:
    theta1 = np.array([18, -1])
    plot_with_loss(x, y, theta1)
    # Output:

    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    # Example 3:
    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)
    # Output:

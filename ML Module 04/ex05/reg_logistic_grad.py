import numpy as np
import math


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """

    return (1 / (1 + math.e ** (-x)))


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three arrayArgs:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of dimesion m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[0] != y.shape[0] or x.shape[1] != theta.shape[0] - 1:
        return None

    m = x.shape[0]  # Number of samples

    # Compute the predicted probabilities using the sigmoid function
    y_hat = sigmoid_(np.dot(x, theta[1:]) + theta[0])

    # Compute the gradient vector
    gradient = np.zeros_like(theta)
    gradient[0] = np.sum(y_hat - y) / m

    for j in range(1, theta.shape[0]):
        gradient[j] = (np.sum((y_hat - y) * x[:, j - 1].reshape(-1, 1)) +
                       (lambda_ * theta[j])) / m
    return gradient


def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arrArgs:
    y: has to be a numpy.ndarray, a vector of shape m * 1.
    x: has to be a numpy.ndarray, a matrix of shape m * n.
    theta: has to be a numpy.ndarray, a vector of shape n * 1.
    lambda_: has to be a float.
    Returns:
    A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
    None if y, x, or theta are empty numpy.ndarray.
    None if y, x or theta does not share compatibles shapes.
    Raises:
    This function should not raise any Exception.
    """
    if x.size == 0 or y.size == 0 or theta.size == 0 or x.shape[0] != y.shape[0] or x.shape[1] != theta.shape[0] - 1:
        return None

    m = x.shape[0]  # Number of samples

    # Add bias term to the feature matrix
    x_with_bias = np.c_[np.ones((m, 1)), x]

    # Compute the predicted probabilities using the sigmoid function
    y_hat = sigmoid_(np.dot(x_with_bias, theta))

    # Compute the gradient vector
    gradient = np.dot(x_with_bias.T, y_hat - y) / m
    gradient[1:] += (lambda_ / m) * theta[1:]

    return gradient


if __name__ == "__main__":
    x = np.array([[0, 2, 3, 4],
                  [2, 4, 5, 5],
                  [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    theta = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    # Example 1.1:
    print(reg_logistic_grad(y, x, theta, 1))
    # # Output:
    # array([[-0.55711039],
    #        [-1.40334809],
    #        [-1.91756886],
    #        [-2.56737958],
    #        [-3.03924017]])
    # Example 1.2:
    print(vec_reg_logistic_grad(y, x, theta, 1))
    # # Output:
    # array([[-0.55711039],
    #        [-1.40334809],
    #        [-1.91756886],
    #        [-2.56737958],
    #        [-3.03924017]])
    # Example 2.1:
    print(reg_logistic_grad(y, x, theta, 0.5))
    # # Output:
    # array([[-0.55711039],
    #        [-1.15334809],
    #        [-1.96756886],
    #        [-2.33404624],
    #        [-3.15590684]])
    # Example 2.2:
    print(vec_reg_logistic_grad(y, x, theta, 0.5))
    # # Output:
    # array([[-0.55711039],
    #        [-1.15334809],
    #        [-1.96756886],
    #        [-2.33404624],
    #        [-3.15590684]])
    # Example 3.1:
    print(reg_logistic_grad(y, x, theta, 0.0))
    # # Output:
    # array([[-0.55711039],
    #        [-0.90334809],
    #        [-2.01756886],
    #        [-2.10071291],
    #        [-3.27257351]])
    # Example 3.2:
    print(vec_reg_logistic_grad(y, x, theta, 0.0))
    # # Output:
    # array([[-0.55711039],
    #        [-0.90334809],
    #        [-2.01756886],
    #        [-2.10071291],
    #        [-3.27257351]])

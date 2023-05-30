import numpy as np


def reg_linear_grad(y, x, theta, lambda_):
    # if type(x) != np.ndarray or type(y) != np.ndarray or type(theta) != np.ndarray:
    #     return None
    # if x.size == 0 or y.size == 0 or theta.size == 0:
    #     return None
    # if x.shape[0] != y.shape[0] or theta.shape != (2, 1):
    #     return None

    m = x.shape[0]
    n = x.shape[1]
    gradient = np.zeros((n + 1, 1))
    X = np.hstack((np.ones((m, 1)), x))

    for i in range(m):
        xi = X[i].reshape(n + 1, 1).astype(float)
        yi = y[i]
        hypothesis = np.dot(theta.T, xi)[0, 0]
        gradient += (hypothesis - yi) * xi

    gradient /= m
    gradient[1:] += (lambda_ / m) * theta[1:]

    return gradient


def vec_reg_linear_grad(y, x, theta, lambda_):
    m = x.shape[0]
    X = np.hstack((np.ones((m, 1)), x))
    h_theta = np.dot(X, theta)

    gradient = (1 / m) * np.dot(X.T, h_theta - y)
    gradient[1:] += (lambda_ / m) * theta[1:]

    return gradient


if __name__ == "__main__":
    x = np.array([
        [-6, -7, -9],
        [13, -2, 14],
        [-7, 14, -1],
        [-8, -4, 6],
        [-5, -9, 6],
        [1, -5, 11],
        [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    theta = np.array([[7.01], [3], [10.5], [-6]])
    # Example 1.1:
    print(reg_linear_grad(y, x, theta, 1))
    # # Output:
    # array([[-60.99],
    #        [-195.64714286],
    #        [863.46571429],
    #        [-644.52142857]])
    # Example 1.2:
    print(vec_reg_linear_grad(y, x, theta, 1))
    # # Output:
    # array([[-60.99],
    #        [-195.64714286],
    #        [863.46571429],
    #        [-644.52142857]])
    # Example 2.1:
    print(reg_linear_grad(y, x, theta, 0.5))
    # # Output:
    # array([[-60.99],
    #        [-195.86142857],
    #        [862.71571429],
    #        [-644.09285714]])
    # Example 2.2:
    print(vec_reg_linear_grad(y, x, theta, 0.5))
    # # Output:
    # array([[-60.99],
    #        [-195.86142857],
    #        [862.71571429],
    #        [-644.09285714]])
    # Example 3.1:
    print(reg_linear_grad(y, x, theta, 0.0))
    # # Output:
    # array([[-60.99],
    #        [-196.07571429],
    #        [861.96571429],
    #        [-643.66428571]])
    # Example 3.2:
    print(vec_reg_linear_grad(y, x, theta, 0.0))
    # # Output:
    # array([[-60.99],
    #        [-196.07571429],
    #        [861.96571429],
    #        [-643.66428571]])

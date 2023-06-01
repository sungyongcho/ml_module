import numpy as np
import math


class MyLogisticRegression():
    """
    Description:
    My personal logistic regression to classify things.
    """

    supported_penalties = ['l2']  # Only 'l2' penalty is considered

    def __init__(self, theta, alpha=0.001, max_iter=1000, penalty='l2', lambda_=1.0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta
        self.penalty = penalty
        self.lambda_ = lambda_ if penalty in self.supported_penalties else 0

    def sigmoid_(self, x):
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

        return 1 / (1 + np.exp(-x))

    def predict_(self, x):
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

        return 1 / (1 + np.exp(-np.dot(x_prime, self.theta)))

    def loss_elem_(self, y_hat, eps=1e-15):
        return np.clip(y_hat, eps, 1 - eps)

    def loss_(self, y, y_hat, eps=1e-15):
        """
        Computes the logistic loss value.
        Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
        Returns:
        The logistic loss value as a float.
        None on any error.
        Raises:
        This function should not raise any Exception.
        """

        m = y.shape[0]  # Number of samples

        y_hat_update = self.loss_elem_(y_hat, eps)

        # Compute the logistic loss
        loss = -np.sum(y * np.log(y_hat_update) + (1 - y)
                       * np.log(1 - y_hat_update)) / m

        if self.penalty == 'l2':
            regularization_term = self.lambda_ / \
                (2 * m) * np.sum(np.square(self.theta[1:]))
            loss += regularization_term

        return loss

    def gradient_(self, x, y):
        """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible shapes.
        Raises:
        This function should not raise any Exception.
        """
        # Check if the inputs are non-empty and have compatible shapes
        if x.size == 0 or y.size == 0 or self.theta.size == 0 or \
                x.shape[0] != y.shape[0] or x.shape[1] != self.theta.shape[0] - 1:
            return None

        m = x.shape[0]  # Number of samples

        # Add bias term to the feature matrix
        x_with_bias = np.c_[np.ones((m, 1)), x]

        # Compute the predicted probabilities using the sigmoid function
        y_hat = self.sigmoid_(np.dot(x_with_bias, self.theta))

        # Compute the gradient vector
        gradient = np.dot(x_with_bias.T, y_hat - y) / m

        if self.penalty == 'l2':
            gradient[1:] += (self.lambda_ / m) * self.theta[1:]

        return gradient

    def fit_(self, x, y):
        for i in range(self.max_iter):
            gradient_update = self.gradient_(x, y)
            if gradient_update is None:
                return None
            self.theta = self.theta.astype(np.float64)
            # Update theta using the mean gradient
            self.theta -= self.alpha * \
                gradient_update.mean(axis=1, keepdims=True)
            if i % 10000 == 0:
                print(i, "th:", self.theta.flatten())
        return self.theta


def ex1():
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas, lambda_=0.0)
    # Example 0:
    y_hat = mylr.predict_(X)
    print(y_hat)  # Output: array([[0.99930437], [1. ], [1. ]])

    # Example 1:
    # print(np.mean(mylr.loss_elem_(Y, y_hat)))
    print(mylr.loss_(Y, y_hat))  # Output: 11.513157421577004

    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)
    # Output: array([[ 2.11826435] [ 0.10154334] [ 6.43942899] [-5.10817488] [ 0.6212541 ]])
    # Example 3:
    y_hat = mylr.predict_(X)
    print(y_hat)  # Output: array([[0.57606717] [0.68599807] [0.06562156]])

    # Example 4:
    # print(np.mean(mylr.loss_elem_(Y, y_hat)))
    print(mylr.loss_(Y, y_hat))  # Output: 1.4779126923052268


def ex2():
    thetas = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    # Example 1:
    model1 = MyLogisticRegression(thetas, lambda_=5.0)
    print(model1.penalty)  # Output ’l2’
    print(model1.lambda_)  # Output 5.0

    # Example 2:
    model2 = MyLogisticRegression(thetas, penalty=None)
    print(model2.penalty)  # Output None
    print(model2.lambda_)  # Output 0.0

    # Example 3:
    model3 = MyLogisticRegression(thetas, penalty=None, lambda_=2.0)
    print(model3.penalty)  # Output None
    print(model3.lambda_)  # Output 0.0


def ex3():
    x = np.array([[0, 2, 3, 4],
                  [2, 4, 5, 5],
                  [1, 3, 2, 7]])
    y = np.array([[0], [1], [1]])
    thetas = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

    model1 = MyLogisticRegression(thetas, lambda_=0.0)

    # Example 1.2:
    # Output: array([[-0.55711039], [-1.40334809], [-1.91756886], [-2.56737958], [-3.03924017]])
    model1.lambda_ = 1.0
    print(model1.gradient_(x, y))
    # Example 2.2:
    # Output: array([[-0.55711039], [-1.15334809], [-1.96756886], [-2.33404624], [-3.15590684]])
    model1.lambda_ = 0.5
    print(model1.gradient_(x, y))
    # Example 3.2:
    # Output: array([[-0.55711039], [-0.90334809], [-2.01756886], [-2.10071291], [-3.27257351]])
    model1.lambda_ = 0.0
    print(model1.gradient_(x, y))


if __name__ == "__main__":
    ex1()
    ex2()
    ex3()

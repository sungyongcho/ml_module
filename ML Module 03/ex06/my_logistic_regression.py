import numpy as np
import math


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """

    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

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

        return (1 / (1 + math.e ** (-x)))

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

        return (1 / (1 + math.e ** (-np.dot(x_prime, self.theta))))

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

        return loss

    def gradient_(self, x, y):
        """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compArgs:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
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

        return gradient

    def fit_(self, x, y):
        for i in range(self.max_iter):
            gradient_update = self.gradient_(x, y)
            if gradient_update is None:
                return None
            self.theta = self.theta.astype(np.float64)
            self.theta -= self.alpha * gradient_update
            if (i % 10000 == 0):
                print(i, "th:", self.theta.flatten())
        return self.theta


if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
    mylr = MyLogisticRegression(thetas)
    # Example 0:
    print(mylr.predict_(X))
    # # Output:
    # array([[0.99930437],
    #        [1.],
    #        [1.]])

    # Example 1:
    # print(np.mean(mylr.loss_elem_(Y, y_hat)))
    y_hat = mylr.predict_(X)
    print(mylr.loss_(Y, y_hat))
    # print(mylr.loss_(X, Y))
    # # Output:
    # 11.513157421577004
    # Example 2:
    mylr.fit_(X, Y)
    print(mylr.theta)
    # # # Output:
    # # array([[2.11826435]
    # #        [0.10154334]
    # #        [6.43942899]
    # #        [-5.10817488]
    # #        [0.6212541]])
    # # Example 3:
    print(mylr.predict_(X))
    # # # Output:
    # # array([[0.57606717]
    # #        [0.68599807]
    # #        [0.06562156]])
    # # Example 4:
    y_hat = mylr.predict_(X)
    print(mylr.loss_(Y, y_hat))
    # # # Output:
    # # 1.4779126923052268

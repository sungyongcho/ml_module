import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = thetas

    def simple_gradient(self, x, y, theta):
        """Computes a gradient vector from three non-empty numpy.array, without any for loop.
        The three arrays must have compatible shapes.
        Args:
        x: has to be a numpy.array, a matrix of shape m * 1.
        y: has to be a numpy.array, a vector of shape m * 1.
        theta: has to be a numpy.array, a 2 * 1 vector.
        Return:
        The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
        None if x, y, or theta is an empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        if x.size == 0 or y.size == 0 or theta.size == 0:
            return None
        if x.shape[0] != y.shape[0] or theta.shape != (2, 1):
            return None

        m = x.shape[0]
        xi = np.hstack((np.ones((m, 1)), x))
        hypothesis = np.dot(xi, theta)
        gradient = np.dot(xi.T, hypothesis - y) / m

        return gradient

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
        Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exception.
        """
        theta_0 = self.thetas[0].astype('float64')
        theta_1 = self.thetas[1].astype('float64')
        for i in range(self.max_iter):
            gradient = self.simple_gradient(x, y, self.thetas)
            if gradient is None:
                return None
            theta_0 -= self.alpha * gradient[0]
            theta_1 -= self.alpha * gradient[1]
            self.thetas = np.array([theta_0, theta_1])
            if (i % 10000 == 0):
                print(i, "th: [", theta_0, ", ", theta_1, "]")

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
        Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exceptions.
        """

        # if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        #     print('a')
        #     return None

        # if x.ndim != 1 or x.size == 0 or theta.shape != (2, 1):
        #     print('b')
        #     return None

        X = np.column_stack((np.ones(len(x)), x))
        y_hat = np.dot(X, self.thetas)

        return y_hat

    def loss_elem_(self, y, y_hat):
        """
        Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """
        a = y_hat - y
        return (a ** 2)
        # print(np.sqrt(a))

    def loss_(self, y, y_hat):
        """
        Description:
        Calculates the value of loss function.
        Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
        Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
        Raises:
        This function should not raise any Exception.
        """

        a = self.loss_elem_(y, y_hat)

        return np.sum(a)/len(a) / 2

    def mse_elem(self, y, y_hat):
        a = y_hat - y
        return (a ** 2)

    def mse_(self, y, y_hat):
        """
        Description:
        Calculate the MSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.array, a vector of dimension m * 1.
        y_hat: has to be a numpy.array, a vector of dimension m * 1.
        Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        if y.shape != y_hat.shape:
            return None
        return np.mean(np.square(y_hat - y))

    def plot_regression(self, x, y, y_hat):

        plt.grid()

        plt.xlabel("Quantity of blue pill (in micrograms)")
        plt.ylabel("Space driving score")

        plt.plot(x, y_hat, "--X", color="lime", linewidth=2, label="S$_{predict}$(pills)")
        plt.plot(x, y, "o", color="cyan", label="S$_{true}$(pills)")

        plt.legend(bbox_to_anchor=(0, 1, 1, 0), loc="lower left", ncol=2, frameon=False)
        plt.show()

    def plot_cost(self, x, y):
        plt.xlabel(r"$\theta_1$")
        plt.ylabel("cost function J$(\\theta_0, \\theta_1)$")
        plt.grid()

        npoints = 100
        thetas_0 = np.linspace(80, 100, 6)
        thetas_1 = np.linspace(-15, -4, npoints)
        for t0 in thetas_0:
            self.thetas[0][0] = t0

            y_cost = [0] * npoints
            for i, t1 in enumerate(thetas_1):
                self.thetas[1][0] = t1
                y_hat = self.predict_(x)
                y_cost[i] = self.mse_(y, y_hat)
            plt.plot(thetas_1, y_cost, label=f"J$(\\theta_0={t0}, \\theta_1)$")

        plt.ylim([10, 150])
        plt.legend(loc="lower right")
        plt.show()

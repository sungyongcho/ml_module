import numpy as np


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


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [
        31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [
        45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLinearRegression(np.array([[2], [0.7]]))
    # Example 0.0:
    y_hat = lr1.predict_(x)
    print(y_hat)
    # # Output:
    # array([[10.74695094],
    #     [17.05055804],
    #     [24.08691674],
    #     [36.24020866],
    #     [42.25621131]])

    # Example 0.1:
    print(lr1.loss_elem_(y, y_hat))
    # # Output:
    # array([[710.45867381],
    # [364.68645485],
    # [469.96221651],
    # [108.97553412],
    # [299.37111101]])
    # Example 0.2:
    print(lr1.loss_(y, y_hat))
    # # Output:
    # 195.34539903032385
    # # Example 1.0:
    lr2 = MyLinearRegression(np.array([[1], [1]]), 5e-8, 1500000)
    lr2.fit_(x, y)
    print(lr2.thetas)
    # # Output:
    # array([[1.40709365],
    # [1.1150909 ]])
    # Example 1.1:
    y_hat = lr2.predict_(x)
    print(y_hat)
    # # Output:
    # array([[15.3408728 ],
    # [25.38243697],
    # [36.59126492],
    # [55.95130097],
    # [65.53471499]])
    # Example 1.2:
    print(lr2.loss_elem_(y, y_hat))
    # # Output:
    # array([[486.66604863],
    # [115.88278416],
    # [ 84.16711596],
    # [ 85.96919719],
    # [ 35.71448348]])
    # Example 1.3:
    print(lr2.loss_(y, y_hat))
    # # Output:
    # 80.83996294128525

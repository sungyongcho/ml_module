from mylinearregression import MyLinearRegression
import numpy as np


class MyRidge(MyLinearRegression):
    def __init__(self, thetas, alpha=0.001, max_iter=1000, lambda_=1):
        super().__init__(thetas, alpha, max_iter)
        self.lambda_ = lambda_

    def get_params_(self):
        return self.__dict__

    def set_params_(self, thetas=None, alpha=None, max_iter=None, lambda_=None):
        if thetas is not None:
            self.thetas = thetas

        if alpha is not None:
            self.alpha = alpha

        if max_iter is not None:
            self.max_iter = max_iter

        if lambda_ is not None:
            self.lambda_ = lambda_

    def loss_elem_(self, y, y_hat):
        return np.square(y_hat - y)

    def loss_(self, y, y_hat):
        if y.size == 0 or y_hat.size == 0 or self.thetas.size == 0:
            return None

        if y.shape != y_hat.shape:
            return None

        loss_elements = self.loss_elem_(y, y_hat)
        loss = np.sum(loss_elements)

        regularization_term = self.lambda_ * np.dot(self.thetas[1:].T, self.thetas[1:])

        return float(1 / (2 * len(y)) * (loss + regularization_term))

    def gradient_(self, x, y):
        m = x.shape[0]
        n = x.shape[1]
        gradient = np.zeros((n + 1, 1))
        X = np.hstack((np.ones((m, 1)), x))

        for i in range(m):
            xi = X[i].reshape(n + 1, 1).astype(float)
            yi = y[i]
            hypothesis = np.dot(self.thetas.T, xi)[0, 0]
            gradient += (hypothesis - yi) * xi

        gradient /= m
        gradient[1:] += (self.lambda_ / m) * self.thetas[1:]

        return gradient

    def fit_(self, x, y):
        m = len(y)  # Number of training examples
        n = x.shape[1]  # Number of features

        # if (x.shape[0] != m) or (self.the.shape[0] != (n + 1)):
        #     return None
        for i in range(self.max_iter):
            # Use the gradient_ method from MyRidge class
            gradient_update = self.gradient_(x, y)
            if gradient_update is None:
                return None
            self.thetas = self.thetas.astype(np.float64)

            # Regularization term
            regularization_term = (self.lambda_ / m) * self.thetas

            # Update thetas with regularization
            self.thetas -= self.alpha * (gradient_update + regularization_term)

            if (i % 10000 == 0):
                print(i, "th:", self.thetas.flatten())
        return self.thetas



if __name__ == "__main__":

    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyRidge(np.array([[1.], [1.], [1.], [1.], [1]]), lambda_=0.0)
    # print("mylr.thetas:", mylr.thetas)

    # Example 0:
    y_hat = mylr.predict_(X)  # Output: array([[8.], [48.], [323.]])
    print(y_hat)

    # Example 1:
    print(mylr.loss_elem_(Y, y_hat))  # Output: array([[225.], [0.], [11025.]])

    # Example 2:
    print(mylr.loss_(Y, y_hat))  # Output: 1875.0

    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    # Output: array([[18.188..], [2.767..], [-0.374..], [1.392..], [0.017..]])
    print(mylr.thetas)

    # Example 4:
    # Output: array([[23.417..], [47.489..], [218.065...]])
    y_hat = mylr.predict_(X)
    print(y_hat)

    # Example 5:
    # Output: array([[0.174..], [0.260..], [0.004..]])
    print(mylr.loss_elem_(Y, y_hat))

    # Example 6:
    print(mylr.loss_(Y, y_hat))  # Output: 0.0732..
    mylr.set_params_(lambda_=2.0)
    print(mylr.get_params_())

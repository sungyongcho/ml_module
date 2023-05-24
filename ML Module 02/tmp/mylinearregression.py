import numpy as np

class MyLinearRegression:
    def __init__(self, thetas, alpha=0.01, max_iter=1000):
        self.thetas = thetas
        self.alpha = alpha
        self.max_iter = max_iter

    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def fit_(self, X, y):
        X = self.add_intercept(X)
        m = X.shape[0]
        n = X.shape[1]

        for i in range(self.max_iter):
            h = np.dot(X, self.thetas)
            gradients = np.dot(X.T, (h - y)) / m
            self.thetas -= self.alpha * gradients
            if i % 100 == 0:
                print(i, "th:", self.thetas.flatten())
            if np.isnan(self.thetas).any():
                print("NaN values encountered. Exiting the loop.")
                break

        return self.thetas


    def predict_(self, X):
        X = self.add_intercept(X)
        return np.dot(X, self.thetas)

import numpy as np


def loss_elem_(y, y_hat):
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


def loss_(y, y_hat):
    """Computes the mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Return:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.array.
    None if y and y_hat does not share the same dimensions.
    None if y or y_hat is not of expected type.
    Raises:
    This function should not raise any Exception.
    """

    a = loss_elem_(y, y_hat)

    return np.sum(a)/len(a) / 2


if __name__ == "__main__":
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # Example 1:
    print(loss_(X, Y))
    # # Output:
    # 2.142857142857143
    # Example 2:
    print(loss_(X, X))
    # # Output
    # 0.0

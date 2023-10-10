import numpy as np
from prediction import predict_


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

    return a**2
    # print(np.sqrt(a))


def loss_(y, y_hat):
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

    a = loss_elem_(y, y_hat)

    return np.sum(a) / len(a) / 2


if __name__ == "__main__":
    x1 = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    theta1 = np.array([[2.0], [4.0]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.0], [7.0], [12.0], [17.0], [22.0]])
    print(loss_elem_(y1, y_hat1))
    # Output:
    # array([[0.], [1], [4], [9], [16]])

    # Example 2:
    print(loss_(y1, y_hat1))
    # # Output:
    # 3.0

    x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
    theta2 = np.array([[0.0], [1.0]]).reshape(-1, 1)
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
    # Example 3:
    print(loss_(y2, y_hat2))
    # # Output:
    # 2.142857142857143
    # Example 4:
    print(loss_(y2, y2))
    # # Output:
    # 0.0

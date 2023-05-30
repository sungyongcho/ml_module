import numpy as np


def data_spliter(x, y, train_proportion, val_proportion):
    """
    Shuffles and splits the dataset (given by x and y) into a training, cross-validation, and test set,
    while respecting the given proportions of examples to be kept in each set.

    Args:
    x: A numpy array, a matrix of dimension m * n.
    y: A numpy array, a vector of dimension m * 1.
    train_proportion: A float, the proportion of examples to be kept in the training set.
    val_proportion: A float, the proportion of examples to be kept in the cross-validation set.

    Returns:
    (x_train, x_val, x_test, y_train, y_val, y_test) as a tuple of numpy arrays.

    Raises:
    This function should not raise any exceptions.
    """

    # Shuffle the indices
    indices = np.random.permutation(x.shape[0])

    # Calculate the split indices
    train_split = int(train_proportion * x.shape[0])
    val_split = int((train_proportion + val_proportion) * x.shape[0])

    # Split the data into training, cross-validation, and test sets
    x_train = x[indices[:train_split]]
    x_val = x[indices[train_split:val_split]]
    x_test = x[indices[val_split:]]

    y_train = y[indices[:train_split]]
    y_val = y[indices[train_split:val_split]]
    y_test = y[indices[val_split:]]

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    # Example 1:
    print(data_spliter(x1, y, 0.8))
    # # # Output:
    # # it's random
    # # (array([ 1, 59, 42, 300]), array([10]), array([0, 0, 1, 0]), array([1]))
    # # Example 2:
    print(data_spliter(x1, y, 0.5))
    # # # Output:
    # # (array([59, 10]), array([ 1, 300, 42]), array([0, 1]), array([0, 0, 1]))
    x2 = np.array([[1, 42],
                   [300, 10],
                   [59, 1],
                   [300, 59],
                   [10, 42]])
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    # # Example 3:
    print(data_spliter(x2, y, 0.8))
    # # # Output:
    # # (array([[ 10, 42],
    # # [300, 59],
    # # [ 59, 1],
    # # [300, 10]]),
    # # array([[ 1, 42]]),
    # # array([0, 1, 0, 1]),
    # # array([0]))
    # # Example 4:
    print(data_spliter(x2, y, 0.5))
    # # Output:
    # # (array([[59, 1],
    # # [10, 42]]),
    # # array([[300, 10],
    # # [300, 59],
    # # [ 1, 42]]),
    # # array([0, 0]),
    # # array([1, 1, 0]))

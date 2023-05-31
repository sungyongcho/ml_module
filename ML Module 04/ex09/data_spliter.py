import numpy as np


def data_spliter(x, y, train_proportion, val_proportion=None, fix='n'):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible dimensions.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if fix == 'y':
        np.random.seed(1)  # Set the random seed for consistent shuffling

    if val_proportion is not None:

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
    # else
    indices = np.random.permutation(x.shape[0])
    split_index = int(train_proportion * x.shape[0])

    x_train = np.array(x[indices[:split_index]])
    x_test = np.array(x[indices[split_index:]])
    y_train = np.array(y[indices[:split_index]])
    y_test = np.array(y[indices[split_index:]])

    return x_train, x_test, y_train, y_test

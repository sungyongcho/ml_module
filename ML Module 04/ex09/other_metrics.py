import numpy as np


def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    # Accuracy Score = (Number of Correct Predictions) / (Total Number of Samples)

    # Check if the inputs have compatible shapes
    if y.shape != y_hat.shape:
        return None

    # Count the number of correct predictions
    correct_predictions = np.sum(y == y_hat)

    # Calculate the accuracy score
    accuracy = correct_predictions / len(y)

    return accuracy


def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    # Precision = True Positives / (True Positives + False Positives)

    # if y.shape != y_hat.shape:
    #     return None

    # Count true positives and false positives
    true_positives = np.sum((y_hat == pos_label) & (y == pos_label))
    false_positives = np.sum((y_hat == pos_label) & (y != pos_label))

    # Compute precision
    precision = true_positives / (true_positives + false_positives)

    return precision


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    # Recall = True Positives / (True Positives + False Negatives)

    if y.shape != y_hat.shape:
        return None

    true_positives = np.sum((y_hat == pos_label) & (y == pos_label))
    false_negatives = np.sum((y_hat != pos_label) & (y == pos_label))

    recall = true_positives / (true_positives + false_negatives)
    return recall


def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """

    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)

    # print(y.shape, y_hat.shape)
    if y.shape != y_hat.shape:
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        if len(y_hat.shape) == 1:
            y_hat = y_hat.reshape(-1, 1)
    # print(y.shape, y_hat.shape)

    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

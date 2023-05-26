import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

    if y.shape != y_hat.shape:
        return None

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

    if y.shape != y_hat.shape:
        return None

    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)

    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def ex1():
    # Example 1:
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # ## Output:
    # 0.5
    # sklearn implementation
    print(accuracy_score(y, y_hat))
    # ## Output:
    # 0.5
    # Precision
    # your implementation
    print(precision_score_(y, y_hat))
    # ## Output:
    # 0.4
    # sklearn implementation
    print(precision_score(y, y_hat))
    # ## Output:
    # 0.4
    # Recall
    # your implementation
    print(recall_score_(y, y_hat))
    # ## Output:
    # 0.6666666666666666
    # sklearn implementation
    print(recall_score(y, y_hat))
    # ## Output:
    # 0.6666666666666666
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat))
    # ## Output:
    # 0.5
    # sklearn implementation
    print(f1_score(y, y_hat))
    # ## Output:
    # 0.5


def ex2():
    # Example 2:
    y_hat = np.array(['norminet', 'dog', 'norminet',
                     'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                 'dog', 'norminet', 'dog', 'norminet'])
    # Accuracy
    # your implementation
    print(accuracy_score_(y, y_hat))
    # # Output:
    # 0.625
    # sklearn implementation
    print(accuracy_score(y, y_hat))
    # # Output:
    # 0.625
    # Precision
    # your implementation
    print(precision_score_(y, y_hat, pos_label='dog'))
    # # Output:
    # 0.6
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='dog'))
    # # Output:
    # 0.6
    # Recall
    # your implementation
    print(recall_score_(y, y_hat, pos_label='dog'))
    # # Output:
    # 0.75
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='dog'))
    # # Output:
    # 0.75
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='dog'))
    # # Output:
    # 0.6666666666666665
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='dog'))
    # # Output:
    # 0.6666666666666665


def ex3():
    # Example 3:
    y_hat = np.array(['norminet', 'dog', 'norminet',
                     'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet',
                 'dog', 'norminet', 'dog', 'norminet'])
    # Precision
    # your implementation
    print(precision_score_(y, y_hat, pos_label='norminet'))
    # ## Output:
    # 0.6666666666666666
    # sklearn implementation
    print(precision_score(y, y_hat, pos_label='norminet'))
    # ## Output:
    # 0.6666666666666666
    # Recall
    # your implementation
    print(recall_score_(y, y_hat, pos_label='norminet'))
    # ## Output:
    # 0.5
    # sklearn implementation
    print(recall_score(y, y_hat, pos_label='norminet'))
    # ## Output:
    # 0.5
    # F1-score
    # your implementation
    print(f1_score_(y, y_hat, pos_label='norminet'))
    # ## Output:
    # 0.5714285714285715
    # sklearn implementation
    print(f1_score(y, y_hat, pos_label='norminet'))

    # Output:
    # 0.5714285714285715
if __name__ == "__main__":
    print("--------Example 1----------------")
    ex1()
    print("--------Example 1----------------")
    print("--------Example 2----------------")
    ex2()
    print("--------Example 2----------------")
    print("--------Example 3----------------")
    ex3()
    print("--------Example 3----------------")

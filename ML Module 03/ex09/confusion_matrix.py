import numpy as np
from sklearn.metrics import confusion_matrix

# ref: https://towardsdatascience.com/understanding-the-confusion-matrix-and-how-to-implement-it-in-python-319202e0fe4d


def confusion_matrix_(y_true, y_hat, labels=None):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    labels: optional, a list of labels to index the matrix.
    This may be used to reorder or select a subset of labels. (default=None)
    df_option: optional, if set to True the function will return a pandas DataFrame
    instead of a numpy array. (default=False)
    Return:
    The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
    None if any error.
    Raises:
    This function should not raise any Exception.
    """

    if y_true.shape != y_hat.shape:
        return None

    # Get unique labels from y_true and y_hat
    if labels is None:
        labels = np.unique(np.concatenate((y_true, y_hat)))
    print(labels)

    # Calculate the confusion matrix
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_hat == pred_label))

    return cm


if __name__ == "__main__":
    y_hat = np.array([['norminet'], ['dog'], ['norminet'],
                     ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], [
                 'norminet'], ['dog'], ['norminet']])
    # Example 1:
    # your implementation
    print(confusion_matrix_(y, y_hat))
    # ## Output:
    # array([[0 0 0]
    # [0 2 1]
    # [1 0 2]])
    # sklearn implementation
    print(confusion_matrix(y, y_hat))
    # ## Output:
    # array([[0 0 0]
    # [0 2 1]
    # [1 0 2]])
    # Example 2:
    # your implementation
    print(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']))
    # ## Output:
    # array([[2 1]
    # [0 2]])
    # sklearn implementation
    print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
    # ## Output:
    # array([[2 1]
    # [0 2]])

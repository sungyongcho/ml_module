import numpy as np
from TinyStatistician import TinyStatistician as TS
from sklearn.preprocessing import MinMaxScaler


def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x’ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """

    aaa = TS()

    min_x = np.min(x)
    max_x = np.max(x)

    if min_x == max_x:
        return x

    return (x - min_x) / (max_x - min_x)

def eval():
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    scaler = MinMaxScaler()
    X_reshaped = X.reshape((-1, 1))  # Reshape X to a 2D array
    scaler.fit(X_reshaped)
    print("==========x============")
    print("--------scikit-learn-----")
    print(scaler.transform(X_reshaped))
    print("--------mine-----")
    print(minmax(X))
    print("==========x============")

    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    Y_reshaped = Y.reshape((-1, 1))  # Reshape X to a 2D array
    scaler.fit(Y_reshaped)
    print("==========Y============")
    print("--------scikit-learn-----")
    print(scaler.transform(Y_reshaped))
    print("--------mine-----")
    print(minmax(Y))
    print("==========Y============")


if __name__ == "__main__":
    # Example 1:
    X = np.array([0, 15, -9, 7, 12, 3, -21]).reshape((-1, 1))
    # print(minmax(X))
    # # Output:
    # array([0.58333333, 1. , 0.33333333, 0.77777778, 0.91666667,
    # 0.66666667, 0. ])
    # Example 2:
    Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    # print(minmax(Y))
    # # Output:
    # array([0.63636364, 1. , 0.18181818, 0.72727273, 0.93939394,
    # 0.6969697 , 0. ])

    eval()

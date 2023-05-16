import numpy
from TinyStatistician import TinyStatistician as TS


def zscore(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x’ as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
    This function shouldn’t raise any Exception.
    """

    aaa = TS()

    result = (x - aaa.mean(x)) / aaa.std(x)
    return result


if __name__ == "__main__":
    X = numpy.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))
    Y = numpy.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
    print(zscore(Y))
    # print(result)

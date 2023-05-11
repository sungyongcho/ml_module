import numpy as np

from matrix import Vector, Matrix


def pdf():
    print("==================pdf========================")
    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1.shape)
    # # Output:
    # (3, 2)
    print(m1.T())
    # # Output:
    # Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(m1.T().shape)
    # # Output
    # (2, 3)
    m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(m1.shape)
    # # Output:
    # (2, 3)
    print(m1.T())
    # # Output:
    # Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1.T().shape)
    # # Output:
    # (3, 2)
    m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0],
                 [2.0, 3.0],
                 [4.0, 5.0],
                 [6.0, 7.0]])
    print(m1 * m2)
    # # Output:
    # Matrix([[28., 34.], [56., 68.]])
    m1 = Matrix([[0.0, 1.0, 2.0],
                 [0.0, 2.0, 4.0]])
    v1 = Vector([[1], [2], [3]])
    print(m1 * v1)
    # # # Output:
    # # Matrix([[8], [16]])
    # # # Or: Vector([[8], [16]
    v1 = Vector([[1], [2], [3]])
    v2 = Vector([[2], [4], [8]])
    print(v1 + v2)
    # # # Output:
    # # Vector([[3], [6], [11]])
    print("==================pdf========================")


def ex1():
    print("==================ex1========================")
    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1)
    print(m1.shape)  # Output: (3, 2)
    print(m1.T())  # Output: Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(m1.T().shape)  # Output: (2, 3)

    m1 = Matrix([[0., 2., 4.], [1., 3., 5.]])
    print(m1)
    print(m1.shape)  # Output: (2, 3)
    print(m1.T())  # Output: Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1.T().shape)  # Output: (3, 2)
    print("==================ex1========================")


def ex_matrix_by_matrix():
    print("==================ex_matrx_by_matrix========================")
    m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [0.0, 2.0, 4.0, 6.0]])  # (2, 4)

    m2 = Matrix([[0.0, 1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0, 7.0]])  # (2, 4)

    m3 = Matrix([[0.0, 1.0],
                 [2.0, 3.0],
                 [4.0, 5.0],
                 [6.0, 7.0]])  # (4, 2)

    print("numpy:", np.array(m1.data) + np.array(m2.data))
    print("mine:", m1 + m2)
    # Output: Matrix([0.0, 2.0, 4.0, 6.0], [4.0, 7.0, 10.0, 13.0])
    print("numpy:", np.array(m2.data) + np.array(m1.data))
    print("mine:", m2 + m1)
    # Output: Matrix([0.0, 2.0, 4.0, 6.0], [4.0, 7.0, 10.0, 13.0])

    print("numpy:", np.array(m1.data) - np.array(m2.data))
    print("mine:", m1 - m2)
    # Output: Matrix([0.0, 0.0, 0.0, 0.0], [-4.0, -3.0, -2.0, -1.0])
    print("numpy:", np.array(m2.data) - np.array(m1.data))
    print("mine:", m2 - m1)
    # Output: Matrix([0.0, 0.0, 0.0, 0.0], [4.0, 3.0, 2.0, 1.0])

    print("numpy:", np.array(m1.data) / 2)
    print("mine:", m1 / 2)
    # Output: Matrix([0.0, 2.0, 4.0, 6.0], [4.0, 7.0, 10.0, 13.0])

    print("numpy:", np.matmul(np.array(m3.data), np.array(m1.data)))
    print("mine:", m3 * m1)
    # Output: Matrix([[0.0, 2.0, 4.0, 6.0], [0.0, 8.0, 16.0, 24.0], [0.0, 14.0, 28.0, 42.0], [0.0, 20, 40.0, 60.0]]) (4, 4)
    print("numpy:", np.matmul(np.array(m1.data), np.array(m3.data)))
    print("mine:", m1 * m3)
    # Output: Matrix([[28.0, 34.0], [56.0, 68.0]]) (2, 2)
    print("==================ex_matrx_by_matrix========================")


def ex_matrix_by_vector():
    print("==================ex_matrx_by_vector========================")
    m1 = Matrix([[0.0, 1.0, 2.0],
                 [0.0, 2.0, 4.0]])  # (2, 3)

    m2 = Matrix([[0.0, 1.0, 2.0]])  # (1, 3)

    v1 = Vector([[1], [2], [3]])  # (3, 1)

    print("numpy:", np.dot(np.array(m1.data), np.array(v1.data)))
    print("mine:", m1 * v1)
    # Output: Matrix([[8], [16]]) # Or: Vector([[8], [16])
    print("numpy:", np.array(m1.data) * 3)
    print("mine:", m1 * 3)
    print(m1 * 3)
    # Output: Matrix([[0.0, 3.0, 6.0], [0.0, 6.0, 12.0]])
    print("numpy:", 2 * np.array(m1.data))
    print("mine:", 2 * m1)
    # Output: Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0], [0.0, 3.0, 6.0]])
    # print("numpy:", np.array(v1.data) * np.array(m1.data))
    # print("mine:", v1 * m1)
    print("==================ex_matrx_by_vector========================")


def ex_vector_by_vector():
    print("==================ex_vector_by_vector========================")
    v1 = Vector([[1], [2], [3]])  # (3, 1)
    v2 = Vector([[2], [4], [8]])
    print("numpy:", np.array(v1.data) + np.array(v2.data))
    print("mine:", v1 + v2)
    # Output: Vector([[3],[6],[11]])
    print("numpy:", np.array(v2.data) + np.array(v1.data))
    print("mine:", v2 + v1)
    # Output: Vector([[3],[6],[11]])
    # print(v1 + 1) # Output: error
    # print(v2 + "hi") # Output: error
    # print(v2 + "hi") # Output: error
    print("numpy:", np.array(v1.data) - np.array(v2.data))
    print("mine:", v1 - v2)
    # Output: Vector([[-1],[-2],[-5]])
    print("numpy:", np.array(v2.data) - np.array(v1.data))
    print("mine:", v2 - v1)
    # Output: Vector([[1],[2],[5]])
    print("==================ex_vector_by_vector========================")


def ex_vector():
    # when data is a list
    print("\n << when data is a list >>")
    print("\n << Column vector of shape n * 1 >>")
    print(Vector([[0.0], [1.0], [2.0], [3.0]]).data)
    print(Vector([[0.0], [-1.0], [2.0], [3.0]]).data)
    # Vector([[0], [1.0], [2.0], [3.0]])
    # Vector([[0.0, 1.0], [1.0], [2.0], [3.0]])
    # Vector([[], [1.0], [2.0], [3.0]])

    print("\n << Row vector of shape 1 * n >>")
    print(Vector([[0.0, 1.0, 2.0, 3.0]]).data)
    print(Vector([[0.0, -1.0, 2.0, 3.0]]).data)
    print(Vector([[-0.0, 1.0, 2.0, 3.0]]).data)
    # Vector([[0, 1.0, 2.0, 3.0]])
    # Vector([[[], 1.0, 2.0, 3.0]])
    # Vector([1.0, [0.0, 1.0, 2.0, 3.0]])
    # Vector([[]])

    # dot product
    print("\n << dot product >>")
    v1 = Vector([[2.0, 3.0, 1.0]])
    v2 = Vector([[2.0], [3.0], [1.0]])
    # v1 = Vector([[0.0], [1.0], [2.0], [3.0]])
    # v2 = Vector([[2.0], [1.5], [2.25], [4.0]])
    print(v1)
    print(v1.dot(v2))

    # transpose
    print("\n << transpose >>")
    print(v1.T())
    print(v1.T().shape)
    print(v2.T())
    print(v2.T().shape)

    # division by scalar
    v3 = v1 / 2.0
    print(v3.data)
    # v3 = Vector(4) / 0
    # v3 = Vector(4) / None
    # v3 = None / Vector(4)
    # v3 = 3 / Vector(4)

    # addition by scalar
    v = Vector([[0.0], [1.0], [2.0], [3.0]])
    v2 = Vector([[1.0], [1.0], [1.0], [1.0]])
    print((v + v2).data)
    print((v - v2).data != (v2 - v).data)

    # multiplication by scalar
    print(v * 4)
    print(4.0 * v)
    # v * "hello"


if __name__ == "__main__":
    pdf()
    ex1()
    ex_matrix_by_matrix()
    ex_matrix_by_vector()
    ex_vector_by_vector()
    ex_vector()

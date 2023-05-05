from matrix import Matrix
from matrix import Vector

# # mine
# m_1 = Matrix([[1.0, 2.0], [3.0, 4.0]])
# m_2 = Matrix((3, 3))

# print(m_1.shape)
# print(m_1.data)

# print(m_2.shape)
# print(m_2.data)
# # mine

m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
print(m1.shape)
# # Output:
# (3, 2)

m1.T()
# # Output:
# Matrix([[0., 2., 4.], [1., 3., 5.]])

print(m1.T().shape)
# # # Output:
# # (2, 3)


m1 = Matrix([[0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 4.0, 6.0]])

m2 = Matrix([[0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 4.0, 6.0]])

print(m1 + m2)

m3 = Matrix([[9.0, 8.0, 7.0],
            [6.0, 5.0, 4.0],
            [3.0, 2.0, 1.0]])

m4 = Matrix([[1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]])

print(m3 - m4)

print(m3 / 2)

print(2 / m3)

print(m1 * m2)
# # Output:
# Matrix([[28., 34.], [56., 68.]])


# m1 = Matrix([[0.0, 1.0, 2.0],
#              [0.0, 2.0, 4.0]])
# v1 = Vector([[1], [2], [3]])
# m1 * v1
# # Output:
# Matrix([[8], [16]])
# # Or: Vector([[8], [16]

from matrix import Matrix

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

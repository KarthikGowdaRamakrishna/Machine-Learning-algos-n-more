import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

matrix_product = np.dot(A, B)
# Alternatively, you can use the @ operator for matrix multiplication in Python 3.5 and above:
# matrix_product = A @ B

print(matrix_product)

import numpy as np

A = np.random.rand(2, 3, 4)
B = np.random.rand(4, 5)

result = np.dot(A, B)

print(result)

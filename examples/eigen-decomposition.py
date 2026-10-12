# Авторский учебный пример ML Formula Wiki
import numpy as np
matrix = np.array([[2., 0.], [0., 1.]])
values, vectors = np.linalg.eigh(matrix)
print(values)
print(np.allclose(matrix @ vectors, vectors * values))

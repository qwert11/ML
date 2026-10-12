# Авторский учебный пример ML Formula Wiki
import numpy as np
Q = np.array([[1., 0.]])
K = np.array([[1., 0.], [0., 1.]])
scores = Q @ K.T / np.sqrt(K.shape[-1])
print(np.round(scores, 6))  # [1 query, 2 keys]

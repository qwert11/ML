# Авторский учебный пример ML Formula Wiki
import numpy as np
K = np.array([[1., 2.], [2., 4.]])
H = np.eye(len(K)) - np.ones_like(K) / len(K)
centered = H @ K @ H
print(centered)
print(centered.mean(axis=0))

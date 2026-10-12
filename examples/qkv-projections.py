# Авторский учебный пример ML Formula Wiki
import numpy as np
X = np.array([[1., 2.]])  # [число токенов, размерность модели]
Wq = np.eye(2)
Wk = np.array([[0., 1.], [1., 0.]])
Wv = 2 * np.eye(2)
Q, K, V = X @ Wq, X @ Wk, X @ Wv
print('Q:', Q, 'K:', K, 'V:', V)

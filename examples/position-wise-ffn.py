# Авторский учебный пример ML Formula Wiki
import numpy as np
x = np.array([[1., -2.]])
W1, W2 = np.eye(2), np.eye(2)
b1, b2 = np.zeros(2), np.zeros(2)
hidden = np.maximum(0, x @ W1 + b1)
print(hidden @ W2 + b2)

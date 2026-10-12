# Авторский учебный пример ML Formula Wiki
import numpy as np
x = np.array([[1., 2., 3.]])
gamma, beta, epsilon = 1., 0., 1e-5
mean = x.mean(axis=-1, keepdims=True)
variance = x.var(axis=-1, keepdims=True)
normalized = gamma * (x-mean) / np.sqrt(variance+epsilon) + beta
print(np.round(normalized, 6))

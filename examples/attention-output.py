# Авторский учебный пример ML Formula Wiki
import numpy as np
# Входы выбраны так, чтобы softmax(scores) дал веса [0.67, 0.33].
Q = np.array([[1.]])
K = np.log(np.array([[0.67], [0.33]]))
V = np.array([[10., 0.], [0., 20.]])
scores = Q @ K.T / np.sqrt(K.shape[-1])
weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
weights /= weights.sum(axis=-1, keepdims=True)
context = weights @ V
print(weights)
print(context)
assert np.allclose(context, [[6.7, 6.6]])

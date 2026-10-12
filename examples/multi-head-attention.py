# Авторский учебный пример ML Formula Wiki
import numpy as np
heads = [np.array([[1., 2.]]), np.array([[3., 4.]])]
W_output = np.eye(4)
joined = np.concatenate(heads, axis=-1)  # объединяем признаки, не токены
print(joined @ W_output)

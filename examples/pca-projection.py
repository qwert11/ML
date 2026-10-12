# Авторский учебный пример ML Formula Wiki
import numpy as np
x = np.array([2., 3.])  # уже центрированные признаки
W = np.array([[1.], [0.]])
print(x @ W)  # из двух координат в одну

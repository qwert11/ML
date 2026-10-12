# Авторский учебный пример ML Formula Wiki
from math import tanh
x, y, gamma, offset = [1, 2], [2, 1], 0.5, 0
print(round(tanh(gamma * sum(a*b for a, b in zip(x, y)) + offset), 6))

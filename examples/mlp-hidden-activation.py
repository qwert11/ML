# Авторский учебный пример ML Formula Wiki
from math import exp
inputs, weights = [1, 2], [0.5, 1]
z = sum(a*w for a, w in zip(inputs, weights))
print(z, round(1 / (1 + exp(-z)), 6))

# Авторский учебный пример ML Formula Wiki
from math import exp
x, y, gamma = [1, 2], [2, 4], 0.5
distance_squared = sum((a-b)**2 for a, b in zip(x, y))
print(round(exp(-gamma * distance_squared), 6))

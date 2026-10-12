# Авторский учебный пример ML Formula Wiki
from math import exp
sigmoid = lambda z: 1 / (1 + exp(-z))
hidden = sigmoid(2 * 0.5)
output = sigmoid(hidden * 2)
print(round(hidden, 6), round(output, 6))

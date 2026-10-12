# Авторский учебный пример ML Formula Wiki
eta, y, prediction, x, weight = 0.1, 1, -1, 2, 0.5
delta = eta * (y - prediction) * x
print(delta, weight + delta)

# Авторский учебный пример ML Formula Wiki
weight, gradient, eta = 4, 3, 0.1
delta = -eta * gradient  # идём против роста ошибки
print(delta, weight + delta)

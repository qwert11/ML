# Авторский учебный пример ML Formula Wiki
from math import log
y, activation = 1, 0.8
loss = -(y*log(activation) + (1-y)*log(1-activation))
print(round(loss, 6))  # один выход, без регуляризации

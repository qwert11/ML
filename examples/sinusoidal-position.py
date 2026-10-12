# Авторский учебный пример ML Formula Wiki
from math import sin, cos
position, d_model = 1, 4
encoding = []
for i in range(d_model // 2):
    angle = position / 10000**(2*i/d_model)
    encoding.extend([sin(angle), cos(angle)])
print([round(value, 6) for value in encoding])

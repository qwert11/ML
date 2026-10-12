# Авторский учебный пример ML Formula Wiki
y, p, x = 1, 0.8, 2
dp_dw = p * (1 - p) * x
dl_dw = (y / p - (1 - y) / (1 - p)) * dp_dw
print(round(dl_dw, 6))

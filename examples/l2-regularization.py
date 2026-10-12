# Авторский учебный пример ML Formula Wiki
weights, lam = [3, 4], 0.2
penalty = lam / 2 * sum(w*w for w in weights)
print(penalty, 1 / lam)  # штраф и C для этой записи

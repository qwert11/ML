# Авторский учебный пример ML Formula Wiki
loss, weights, lam = 5, [3, 4], 0.1
print(loss + lam * sum(w*w for w in weights))

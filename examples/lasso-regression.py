# Авторский учебный пример ML Formula Wiki
loss, weights, lam = 5, [-2, 3], 0.5
print(loss + lam * sum(abs(w) for w in weights))

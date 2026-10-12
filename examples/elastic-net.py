# Авторский учебный пример ML Formula Wiki
loss, weights, l2, l1 = 4, [1, -2], 0.1, 0.2
print(loss + l2 * sum(w*w for w in weights) + l1 * sum(abs(w) for w in weights))

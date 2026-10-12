# Авторский учебный пример ML Formula Wiki
weight, eta = 0.5, 0.1
y, p, x = [1, 0], [0.8, 0.3], [2, 1]
delta = eta * sum((label - prob) * feature
                  for label, prob, feature in zip(y, p, x))
print(round(delta, 6), round(weight + delta, 6))

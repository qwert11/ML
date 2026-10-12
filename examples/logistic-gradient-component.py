# Авторский учебный пример ML Formula Wiki
y, p, x = [1, 0], [0.8, 0.3], [2, 1]
gradient = -sum((label - prob) * feature
                for label, prob, feature in zip(y, p, x))
print(round(gradient, 6))  # градиент отрицательного log-likelihood

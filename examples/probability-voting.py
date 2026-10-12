# Авторский учебный пример ML Formula Wiki
probabilities = [[0.1, 0.9], [0.4, 0.6]]
scores = [sum(row[i] for row in probabilities) for i in range(2)]
print(scores, max(range(2), key=lambda i: scores[i]))

# Авторский учебный пример ML Formula Wiki
truth, predicted = [1, 0, 1, 1], [1, 0, 0, 1]
error = sum(a != b for a, b in zip(truth, predicted)) / len(truth)
print(error, 1 - error)

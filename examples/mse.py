# Авторский учебный пример ML Formula Wiki
truth, prediction = [2, 4], [3, 2]
print(sum((y-p)**2 for y, p in zip(truth, prediction)) / len(truth))

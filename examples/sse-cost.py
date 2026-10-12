# Авторский учебный пример ML Formula Wiki
y, prediction = [1, 3], [2, 2]
loss = sum((a - b)**2 for a, b in zip(y, prediction)) / 2
print(loss)

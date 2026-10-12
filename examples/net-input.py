# Авторский учебный пример ML Formula Wiki
weights = [-1, 0.5, 1]
x = [1, 2, 3]  # первая единица отвечает за свободный член
z = sum(w * value for w, value in zip(weights, x))
print(z)

# Авторский учебный пример ML Formula Wiki
y, p = [1, 0], [0.8, 0.3]
likelihood = 1.0
for label, probability in zip(y, p):
    likelihood *= probability**label * (1 - probability)**(1 - label)
print(round(likelihood, 6))

# Авторский учебный пример ML Formula Wiki
x, y = [1, 2], [4, 6]
for p in [1, 2]:
    distance = sum(abs(a-b)**p for a, b in zip(x, y))**(1/p)
    print(p, distance)

# Авторский учебный пример ML Formula Wiki
x, y = [1, 2], [2, 1]
phi = lambda v: [v[0]**2, 2**0.5 * v[0]*v[1], v[1]**2]
dot = lambda a, b: sum(x*y for x, y in zip(a, b))
print(round(dot(phi(x), phi(y)), 6), dot(x, y)**2)

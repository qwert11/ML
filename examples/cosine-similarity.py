# Авторский учебный пример ML Formula Wiki
from math import sqrt
u, v = [1, 2], [2, 1]
dot = sum(a*b for a, b in zip(u, v))
norm = lambda x: sqrt(sum(a*a for a in x))
similarity = dot / (norm(u)*norm(v))
print(round(similarity, 6))

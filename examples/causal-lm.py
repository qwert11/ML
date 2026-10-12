# Авторский учебный пример ML Formula Wiki
from math import log, prod
conditional_probabilities = [0.5, 0.4, 0.2]
probability = prod(conditional_probabilities)
loss = -sum(log(p) for p in conditional_probabilities)
print(round(probability, 6), round(loss, 6))
assert abs(loss + log(probability)) < 1e-12

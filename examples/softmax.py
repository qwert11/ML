# Авторский учебный пример ML Formula Wiki
from math import exp
scores = [2, 1, 0]
exponents = [exp(z-max(scores)) for z in scores]  # устойчивый softmax
probabilities = [e/sum(exponents) for e in exponents]
print([round(p, 6) for p in probabilities])
assert abs(sum(probabilities)-1) < 1e-12

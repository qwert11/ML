# Авторский учебный пример ML Formula Wiki
from math import exp
scores = [1., 2.]
exponents = [exp(s-max(scores)) for s in scores]
weights = [x/sum(exponents) for x in exponents]
print([round(w, 6) for w in weights])
assert abs(sum(weights)-1) < 1e-12

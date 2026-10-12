# Авторский учебный пример ML Formula Wiki
from math import log2
probabilities = [0.5, 0.5]
entropy = -sum(p * log2(p) for p in probabilities if p > 0)
print(entropy)  # биты, поскольку логарифм по основанию 2

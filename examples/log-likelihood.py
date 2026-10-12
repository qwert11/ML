# Авторский учебный пример ML Formula Wiki
from math import log
y, p = [1, 0], [0.8, 0.3]
terms = [label * log(prob) + (1 - label) * log(1 - prob)
         for label, prob in zip(y, p)]
print([round(term, 6) for term in terms])
print(round(sum(terms), 6))

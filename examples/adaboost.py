# Авторский учебный пример ML Formula Wiki
from math import log, exp
error = 0.2
alpha = 0.5 * log((1-error) / error)
weights, y, predicted = [0.5, 0.5], [1, -1], [1, 1]
updated = [w * exp(-alpha * target * guess)
           for w, target, guess in zip(weights, y, predicted)]
print(round(alpha, 6))
print([round(w / sum(updated), 6) for w in updated])

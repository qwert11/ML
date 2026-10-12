# Авторский учебный пример ML Formula Wiki
from math import comb
n, error = 3, 0.2
probability = sum(comb(n, k) * error**k * (1-error)**(n-k)
                  for k in range(n//2 + 1, n+1))
print(round(probability, 6))  # предполагаем независимые ошибки

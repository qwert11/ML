# Авторский учебный пример ML Formula Wiki
import numpy as np
x, y = np.array([1., 2., 3.]), np.array([2., 4., 6.])
cov = ((x - x.mean()) * (y - y.mean())).sum() / (len(x) - 1)
print(cov)  # выборочная ковариация: знаменатель n-1

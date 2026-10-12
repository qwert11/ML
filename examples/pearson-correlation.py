# Авторский учебный пример ML Formula Wiki
import numpy as np
x, y = [1, 2, 3], [2, 4, 6]
print(round(float(np.corrcoef(x, y)[0, 1]), 6))

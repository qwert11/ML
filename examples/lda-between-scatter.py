# Авторский учебный пример ML Formula Wiki
import numpy as np
classes = [np.array([[1., 2.], [3., 4.]]), np.array([[6., 3.], [8., 5.]])]
overall = np.concatenate(classes).mean(axis=0)
scatter = np.zeros((2, 2))
for samples in classes:
    delta = samples.mean(axis=0) - overall
    scatter += len(samples) * np.outer(delta, delta)
print(scatter)

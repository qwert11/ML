# Авторский учебный пример ML Formula Wiki
loss = lambda w: w*w
w, eps = 3, 0.01
numeric = (loss(w+eps) - loss(w-eps)) / (2*eps)
analytic = 2*w
print(round(numeric, 6), abs(numeric-analytic) < 1e-9)

# Авторский учебный пример ML Formula Wiki
y, output, hidden, weight = 1, 0.8, 0.5, 2
delta_output = output - y
delta_hidden = weight * delta_output * hidden * (1-hidden)
gradient = hidden * delta_output
print([round(x, 6) for x in [delta_output, delta_hidden, gradient]])

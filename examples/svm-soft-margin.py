# Авторский учебный пример ML Formula Wiki
weights, slack, C = [1, 2], [0, 0.5, 1], 2
objective = sum(w*w for w in weights) / 2 + C * sum(slack)
print(objective)

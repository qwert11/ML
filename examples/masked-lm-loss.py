# Авторский учебный пример ML Formula Wiki
from math import log
correct_token_probabilities = [0.8]  # только позиции, выбранные для MLM
loss = -sum(log(p) for p in correct_token_probabilities)
print(round(loss, 6))

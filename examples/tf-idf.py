# Авторский учебный пример ML Formula Wiki
from math import log
tf, n_documents, document_frequency = 3, 100, 9
idf = log(n_documents / (1 + document_frequency))  # вариант записи в конспекте
print(round(idf, 6), round(tf * idf, 6))

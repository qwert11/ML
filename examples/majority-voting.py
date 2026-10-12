# Авторский учебный пример ML Formula Wiki
from collections import Counter
votes = ['cat', 'dog', 'cat']
print(Counter(votes).most_common(1)[0][0])

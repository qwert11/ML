# Авторский учебный пример ML Formula Wiki
votes, weights = [1, 0], [0.7, 0.3]
scores = {label: sum(w for vote, w in zip(votes, weights) if vote == label)
          for label in set(votes)}
print(max(scores, key=scores.get))

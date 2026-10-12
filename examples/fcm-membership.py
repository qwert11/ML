# Авторский учебный пример ML Formula Wiki
distances, m = [1, 2], 2
# Для нулевого расстояния нужна отдельная ветка; здесь все расстояния > 0.
memberships = [1 / sum((di/dj)**(2/(m-1)) for dj in distances) for di in distances]
print(memberships)

# Авторский учебный пример ML Formula Wiki
parent_impurity = 0.5
children = [(6, 0.25), (4, 0.0)]  # размер и неоднородность
gain = parent_impurity - sum(size / 10 * impurity for size, impurity in children)
print(round(gain, 6))

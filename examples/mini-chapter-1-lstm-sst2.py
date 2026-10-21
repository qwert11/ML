import math

# Тот же «Сокращённый оригинал», что выше, но без Keras: всё, что
# Keras делает внутри своих функций, здесь написано руками.
# Блоки 1–6 идут в том же порядке и под теми же номерами.
# «# Keras: …» — какую строку Keras повторяет блок под этой пометкой.
# Программа печатает каждое число, которое считает. Все они округлены
# до 6 значащих цифр (функция r в блоке 4), и дальше программа считает
# уже с округлённым числом. Поэтому любое число в выводе получается на
# калькуляторе из напечатанных выше по формуле в строке кода рядом.

# ── 1. Данные ──────────────────────────────────────────────
# Keras: sentences = [...]; labels = np.array([1, 0])
# в книге: 67 349 отзывов из SST-2; 1 — хороший, 0 — плохой
sentences = ["very good movie", "bad movie"]
labels = [1, 0]

# ── 2. Словарь: слово → номер ──────────────────────────────
# Keras: words = ["PAD"] + list(...); word2idx = {...}
max_sen_len = max(len(s.split()) for s in sentences)  # самый длинный
# PAD — номер 0: им дополняют короткие отзывы до общей длины
words = ["PAD"] + list(dict.fromkeys(
    [w for s in sentences for w in s.split()]))
word2idx = {w: i for i, w in enumerate(words)}
max_words = max(word2idx.values()) + 1  # слов в словаре = строк в E
print("word2idx:", word2idx)

# ── 3. Отзывы → номера, выравнивание ───────────────────────
# Keras: train = [...]; train_pad = pad_sequences(train, maxlen=...)
train = [[word2idx[w] for w in s.split()] for s in sentences]
# pad_sequences дописывает нули слева: [4, 3] → [0, 4, 3]
train_pad = [[0] * (max_sen_len - len(t)) + t for t in train]
print("train_pad:", train_pad)

# ── 4. Модель: веса слоёв и что каждый слой считает ────────
# Keras: model.add(Embedding(max_words, 2)) + set_weights(...)
# таблица: по строке из 2 чисел на каждое слово словаря
E = [[0.0, 0.0],    # 0 PAD
     [0.2, 0.4],    # 1 very
     [0.9, 0.1],    # 2 good
     [0.1, 0.5],    # 3 movie
     [-0.8, 0.3]]   # 4 bad

# Keras: model.add(LSTM(1)) + set_weights([W, U, b])
# одна ячейка, у неё 4 вентиля — по столбцу на вентиль
#      i    f     c~   o
W = [[0.5, 0.3, 1.0, 0.4],    # от x[0]
     [0.2, -0.1, 0.3, 0.1]]   # от x[1]
U = [0.3, 0.2, 0.5, 0.2]      # от h прошлого слова
b = [0.0, 1.0, 0.0, 0.0]      # у f смещение 1, как в Keras


def r(x):
    # Округляет до 6 значащих цифр; ровно посередине (…5) — вверх, как
    # на калькуляторе. Дальше программа считает уже с округлённым числом,
    # поэтому каждое следующее число сходится из напечатанных. Без этого
    # сеть считала бы с 16 знаками, и последняя напечатанная цифра иногда
    # не сходилась бы с пересчётом. (Keras считает в float32 — ~7 цифр.)
    return float(f"{x:.6g}")


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Что слой LSTM делает с одним словом: вектор слова x
# и память (h, c) после прошлого слова → новая память.
def lstm_step(x, h, c):
    # аргумент каждого вентиля: x·W + h·U + b (столбец W и U на вентиль)
    z = [r(x[0] * W[0][k] + x[1] * W[1][k] + h * U[k] + b[k])
         for k in range(4)]
    print(f"   z_i = {z[0]}  z_f = {z[1]}  z_c = {z[2]}  z_o = {z[3]}")
    i = r(sigmoid(z[0]))        # сколько нового записать
    f = r(sigmoid(z[1]))        # сколько старой памяти оставить
    c_hat = r(math.tanh(z[2]))  # что записать
    o = r(sigmoid(z[3]))        # сколько памяти выдать наружу
    print(f"   i = {i}  f = {f}  c~ = {c_hat}  o = {o}")
    c = r(f * c + i * c_hat)    # новая память
    h = r(o * math.tanh(c))     # выход ячейки
    print(f"   c = {c}  h = {h}")
    return h, c


# Keras: model.add(Dense(1, activation="sigmoid")) + set_weights
# один вес на вход h и смещение; _d — чтобы не путать с b ячейки LSTM
w_d, b_d = 2.0, 0.0

# ── 5. compile: чем мерить ошибку, как менять веса ─────────
# Keras: model.compile(optimizer="rmsprop", loss="binary_crossentropy")
# loss="binary_crossentropy": средняя ошибка по отзывам
def bce(p):
    return -sum(y * math.log(q) + (1 - y) * math.log(1 - q)
                for y, q in zip(labels, p)) / len(p)


# optimizer="rmsprop": его параметры по умолчанию в Keras
lr, rho, eps = 0.001, 0.9, 1e-7

# ── 6. fit: обучение ───────────────────────────────────────
# Keras: model.fit(...) — внутри: прямой проход Embedding → LSTM
# В Keras fit — одна строка: history = model.fit(train_pad, labels,
# epochs=3, batch_size=32). Отсюда и до конца — то, что у неё внутри.
# Прямой проход Keras делает заново каждую эпоху. Но Embedding и LSTM
# заморожены (trainable = False), H во всех эпохах одинаковый — поэтому
# здесь он посчитан один раз, до эпох: каждый отзыв → одно число h.
H = []
for k, seq in enumerate(train_pad):
    print(f"отзыв {k + 1} «{sentences[k]}»")
    h, c = 0.0, 0.0  # перед первым словом памяти нет
    for idx in seq:
        x = E[idx]  # Embedding: номер слова выбирает строку таблицы
        print(f"{words[idx]:>5}: x = {x}  h = {h}  c = {c}")
        h, c = lstm_step(x, h, c)  # шаг LSTM: печатает всё, что посчитал
    H.append(h)  # дальше идёт только h после последнего слова
    print(f"   h_T = {h}  (в Dense пойдёт только оно)")

# Keras: model.fit(...) — внутри: эпохи
# batch_size=32, а отзывов 2 — оба в одном батче: эпоха = один шаг.
# Учится только Dense: градиент назад через ячейку LSTM и таблицу E
# удвоил бы разбор; в книге учатся все слои, тем же правилом цепочки.
v_w = v_b = 0.0  # rmsprop копит средний квадрат градиента; сначала 0
for epoch in range(1, 4):  # epochs=3
    print(f"Epoch {epoch}/3")
    # прямой проход до конца: Dense + sigmoid → вероятность «хороший»
    p = [r(sigmoid(w_d * h + b_d)) for h in H]
    print(f"   p_1 = {p[0]}  p_2 = {p[1]}")
    # metrics=["acc"]: доля отзывов, где p > 0.5 совпало с меткой
    acc = sum((q > 0.5) == y for q, y in zip(p, labels)) / len(p)
    # loss — эту ошибку Keras печатает в строке эпохи
    loss = r(bce(p))
    print(f"   acc = {acc}  loss = {loss}")
    # обратный проход: производные loss по весам Dense
    g_w = r(sum((q - y) * h for q, y, h in zip(p, labels, H)) / len(p))
    g_b = r(sum(q - y for q, y in zip(p, labels)) / len(p))
    print(f"   g_w = {g_w}  g_b = {g_b}")
    # шаг rmsprop: градиент делится на свой средний размер
    v_w = r(rho * v_w + (1 - rho) * g_w ** 2)
    v_b = r(rho * v_b + (1 - rho) * g_b ** 2)
    print(f"   v_w = {v_w}  v_b = {v_b}")
    w_d = r(w_d - lr * g_w / math.sqrt(v_w + eps))
    b_d = r(b_d - lr * g_b / math.sqrt(v_b + eps))
    print(f"   w_d = {w_d}  b_d = {b_d}")
    # так печатает Keras: батч 1 из 1, acc и loss эпохи — посчитаны до шага
    print(f"1/1 - acc: {acc:.4f} - loss: {loss:.4f}")

# Keras: model.layers[2].get_weights() — веса Dense после обучения
print(f"Dense после обучения: w={w_d:.4f} b={b_d:.4f}")

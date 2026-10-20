import math

# Тот же «Сокращённый оригинал», что выше, но без Keras: всё, что
# Keras делает внутри своих функций, здесь написано руками.
# Блоки 1–6 идут в том же порядке и под теми же номерами.
# «# Keras: …» — какую строку Keras повторяет блок под этой пометкой.

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


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


# Что слой LSTM делает с одним словом: вектор слова x
# и память (h, c) после прошлого слова → новая память.
def lstm_step(x, h, c):
    # аргумент каждого вентиля: W·x + U·h + b
    z = [x[0] * W[0][k] + x[1] * W[1][k] + h * U[k] + b[k]
         for k in range(4)]
    i = sigmoid(z[0])        # сколько нового записать
    f = sigmoid(z[1])        # сколько старой памяти оставить
    c_hat = math.tanh(z[2])  # что записать
    o = sigmoid(z[3])        # сколько памяти выдать наружу
    c = f * c + i * c_hat    # новая память
    h = o * math.tanh(c)     # выход ячейки
    return h, c


# Keras: model.add(Dense(1, activation="sigmoid")) + set_weights
# один вес на вход h и смещение
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
for seq in train_pad:
    h, c = 0.0, 0.0  # перед первым словом памяти нет
    for idx in seq:
        # Embedding: E[idx] — строка таблицы; затем шаг LSTM
        h, c = lstm_step(E[idx], h, c)
        print(f"{words[idx]:>5}: h={h:.4f} c={c:.4f}")
    H.append(h)  # дальше идёт только h после последнего слова

# Keras: model.fit(...) — внутри: эпохи
# batch_size=32, а отзывов 2 — оба в одном батче: эпоха = один шаг.
v_w = v_b = 0.0  # rmsprop копит средний квадрат градиента; сначала 0
for epoch in range(1, 4):  # epochs=3
    # прямой проход до конца: Dense + sigmoid → вероятность «хороший»
    p = [sigmoid(w_d * h + b_d) for h in H]
    # metrics=["acc"]: доля отзывов, где p > 0.5 совпало с меткой
    acc = sum((q > 0.5) == y for q, y in zip(p, labels)) / 2
    # loss — эту ошибку Keras печатает в строке эпохи
    loss = bce(p)
    # обратный проход: производные loss по весам Dense
    g_w = sum((q - y) * h for q, y, h in zip(p, labels, H)) / 2
    g_b = sum(q - y for q, y in zip(p, labels)) / 2
    # шаг rmsprop: градиент делится на свой средний размер
    v_w = rho * v_w + (1 - rho) * g_w ** 2
    v_b = rho * v_b + (1 - rho) * g_b ** 2
    w_d -= lr * g_w / math.sqrt(v_w + eps)
    b_d -= lr * g_b / math.sqrt(v_b + eps)
    print(f"Epoch {epoch}/3 - acc: {acc:.4f} - loss: {loss:.4f}")

# Keras: model.layers[2].get_weights() — веса Dense после обучения
print(f"Dense после обучения: w={w_d:.4f} b={b_d:.4f}")

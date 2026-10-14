import numpy as np

# В книге 67 349 отзывов из SST-2/train.tsv, здесь — 2
sentences = ["very good movie", "bad movie"]
labels = np.array([1, 0])   # 1 — позитив, 0 — негатив

max_sen_len = max(len(s.split()) for s in sentences)
# В книге порядок задаёт set() и меняется от запуска к запуску;
# здесь он зафиксирован, чтобы номера слов совпадали с разбором
words = ["PAD", "very", "good", "movie", "bad"]
word2idx = {w: i for i, w in enumerate(words)}
max_words = max(word2idx.values()) + 1
print("word2idx =", word2idx)
print("max_sen_len =", max_sen_len, "max_words =", max_words)

train = [[word2idx[w] for w in s.split()] for s in sentences]
# pad_sequences по умолчанию (padding="pre") дописывает нули СЛЕВА
train_pad = np.array([[0] * (max_sen_len - len(t)) + t for t in train])
print("train =", train)
print("train_pad =", train_pad.tolist())

# Embedding(max_words, 2): таблица 5 × 2 — по строке на слово.
# Keras заполняет её случайно, здесь — круглые числа
E = np.array([[ 0.0, 0.0],   # 0 PAD
              [ 0.2, 0.4],   # 1 very
              [ 0.9, 0.1],   # 2 good
              [ 0.1, 0.5],   # 3 movie
              [-0.8, 0.3]])  # 4 bad
X = E[train_pad[0]]   # строки 1, 2, 3 — «very good movie»
print("X =", X.tolist())

# LSTM(1): одна ячейка. Столбцы — 4 вентиля в порядке Keras:
#                i     f     c~    o
W = np.array([[0.5,  0.3,  1.0,  0.4],    # вес от x[0]
              [0.2, -0.1,  0.3,  0.1]])   # вес от x[1]
U = np.array( [0.3,  0.2,  0.5,  0.2])    # вес от h прошлого шага
b = np.array( [0.0,  1.0,  0.0,  0.0])    # у f смещение 1 (unit_forget_bias)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def lstm_step(x, h, c):
    z = x @ W + h * U + b    # z = x0*W[0] + x1*W[1] + h*U + b -> 4 числа
    i = sigmoid(z[0])        # вход: сколько нового записать
    f = sigmoid(z[1])        # забывание: сколько старой памяти оставить
    c_hat = np.tanh(z[2])    # кандидат: что записать (от -1 до 1)
    o = sigmoid(z[3])        # выход: сколько памяти показать наружу
    c = f * c + i * c_hat    # новая память
    h = o * np.tanh(c)       # выход ячейки
    print("  z =", np.round(z, 4).tolist())
    print(f"  i={i:.4f} f={f:.4f} c_hat={c_hat:.4f} o={o:.4f}")
    print(f"  c={c:.4f} h={h:.4f}")
    return h, c

h, c = 0.0, 0.0   # до первого слова памяти нет
print("t=1 very")
h, c = lstm_step(X[0], h, c)

print("t=2 good")
h, c = lstm_step(X[1], h, c)

print("t=3 movie")
h, c = lstm_step(X[2], h, c)

h2, c2 = 0.0, 0.0
for t, idx in enumerate(train_pad[1], 1):   # [0, 4, 3]
    print(f"t={t} {words[idx]}")
    h2, c2 = lstm_step(E[idx], h2, c2)

H = np.array([h, h2])   # LSTM отдаёт только последнее h каждого отзыва
print("H =", np.round(H, 4).tolist())

w_d, b_d = 2.0, 0.0   # Dense(1): один вес на h и смещение
p = sigmoid(w_d * H + b_d)
print("p =", np.round(p, 4).tolist())

y = labels
L = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
acc = np.mean((p > 0.5) == y)
print(f"loss = {L:.4f}  acc = {acc}")

# Внутри fit TensorFlow считает производные ВСЕХ весов (E, W, U, b, Dense).
# Здесь — только Dense: для пары sigmoid + binary_crossentropy dL/dz = p - y
g_w = np.mean((p - y) * H)
g_b = np.mean(p - y)
print(f"g_w = {g_w:.4f}  g_b = {g_b:.4f}")

lr, rho, eps = 0.001, 0.9, 1e-7          # параметры rmsprop по умолчанию
v_w = rho * 0 + (1 - rho) * g_w ** 2     # среднее квадрата градиента; было 0
v_b = rho * 0 + (1 - rho) * g_b ** 2
w_d = w_d - lr * g_w / (np.sqrt(v_w) + eps)
b_d = b_d - lr * g_b / (np.sqrt(v_b) + eps)

p = sigmoid(w_d * H + b_d)               # снова прямой проход
L = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
print(f"w = {w_d:.4f}  b = {b_d:.4f}")
print("p =", np.round(p, 4).tolist(), f" loss = {L:.4f}")

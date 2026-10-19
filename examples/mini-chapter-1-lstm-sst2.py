import math

# книга: df = pd.read_csv("SST-2/train.tsv", sep="\t")
sentences = ["very good movie", "bad movie"]
labels = [1, 0]

# книга: words = ["PAD"] + list(set(...)); word2idx = {...}
words = ["PAD", "very", "good", "movie", "bad"]
word2idx = {w: i for i, w in enumerate(words)}
max_sen_len = max(len(s.split()) for s in sentences)
max_words = max(word2idx.values()) + 1
print("word2idx:", word2idx)

# книга: train = [...]; train_pad = pad_sequences(train, ...)
train = [[word2idx[w] for w in s.split()] for s in sentences]
train_pad = [[0] * (max_sen_len - len(t)) + t for t in train]
print("train_pad:", train_pad)

# книга: model.add(Embedding(max_words, 32))
E = [[0.0, 0.0],    # PAD
     [0.2, 0.4],    # very
     [0.9, 0.1],    # good
     [0.1, 0.5],    # movie
     [-0.8, 0.3]]   # bad

# книга: model.add(LSTM(32))
#      i    f     c~   o
W = [[0.5, 0.3, 1.0, 0.4],    # от x[0]
     [0.2, -0.1, 0.3, 0.1]]   # от x[1]
U = [0.3, 0.2, 0.5, 0.2]      # от h
b = [0.0, 1.0, 0.0, 0.0]


def sigmoid(z):
    return 1 / (1 + math.exp(-z))


def lstm_step(x, h, c):
    z = [x[0] * W[0][k] + x[1] * W[1][k] + h * U[k] + b[k]
         for k in range(4)]
    i = sigmoid(z[0])
    f = sigmoid(z[1])
    c_hat = math.tanh(z[2])
    o = sigmoid(z[3])
    c = f * c + i * c_hat
    h = o * math.tanh(c)
    return h, c


H = []
for seq in train_pad:
    h, c = 0.0, 0.0
    for idx in seq:
        h, c = lstm_step(E[idx], h, c)
        print(f"{words[idx]:>5}: h={h:.4f} c={c:.4f}")
    H.append(h)

# книга: model.add(Dense(1, activation="sigmoid"))
w_d, b_d = 2.0, 0.0
p = [sigmoid(w_d * h + b_d) for h in H]


# книга: model.compile(optimizer="rmsprop", loss="binary_crossentropy")
def bce(p):
    return -sum(y * math.log(q) + (1 - y) * math.log(1 - q)
                for y, q in zip(labels, p)) / len(p)


loss = bce(p)
print(f"p=[{p[0]:.4f}, {p[1]:.4f}] loss={loss:.4f}")

# книга: model.fit(train_pad, labels, epochs=30, batch_size=32)
# обучаем только Dense; rmsprop: lr=0.001, rho=0.9
v_w = v_b = 0.0
for epoch in range(1, 4):
    g_w = sum((q - y) * h for q, y, h in zip(p, labels, H)) / 2
    g_b = sum(q - y for q, y in zip(p, labels)) / 2
    v_w = 0.9 * v_w + 0.1 * g_w ** 2
    v_b = 0.9 * v_b + 0.1 * g_b ** 2
    w_d -= 0.001 * g_w / (math.sqrt(v_w) + 1e-7)
    b_d -= 0.001 * g_b / (math.sqrt(v_b) + 1e-7)
    p = [sigmoid(w_d * h + b_d) for h in H]
    loss = bce(p)
    print(f"epoch {epoch}: w={w_d:.4f} b={b_d:.4f} loss={loss:.4f}")

import numpy as np

# Мини-версия примера книги: 2 предложения, словарь из 5 слов,
# Embedding(5, 2) → LSTM(1) → Dense(1, sigmoid). Всё считается вручную.

sentences = ["very good movie", "bad movie"]
labels = np.array([1, 0])

# В книге порядок слов задаёт set(); здесь фиксируем его, чтобы индексы не прыгали
words = ["PAD", "very", "good", "movie", "bad"]
word2idx = {w: i for i, w in enumerate(words)}
max_words = len(words)                                   # 5
max_sen_len = max(len(s.split()) for s in sentences)     # 3

train = [[word2idx[w] for w in s.split()] for s in sentences]
# pad_sequences(padding="pre"): нули дописываются слева
train_pad = np.array([[0] * (max_sen_len - len(t)) + t for t in train])
print("train_pad =", train_pad.tolist())

# Keras начинает со случайных весов; здесь — круглые, чтобы считать на бумаге
E = np.array([[0.0, 0.0],    # PAD
              [0.2, 0.4],    # very
              [0.9, 0.1],    # good
              [0.1, 0.5],    # movie
              [-0.8, 0.3]])  # bad          Embedding(max_words, 2)
W = np.array([[0.5, 0.3, 1.0, 0.4],
              [0.2, -0.1, 0.3, 0.1]])  # LSTM kernel: x → вентили [i, f, c̃, o]
U = np.array([0.3, 0.2, 0.5, 0.2])     # recurrent_kernel: h → [i, f, c̃, o]
b = np.array([0.0, 1.0, 0.0, 0.0])     # unit_forget_bias=True: у f смещение 1
w_d, b_d = 2.0, 0.0                    # Dense(1, activation="sigmoid")

sigmoid = lambda z: 1 / (1 + np.exp(-z))

def lstm(seq, show=False):
    h = c = 0.0
    for t, idx in enumerate(seq, 1):
        z = E[idx] @ W + h * U + b
        i, f, o = sigmoid(z[0]), sigmoid(z[1]), sigmoid(z[3])
        c_hat = np.tanh(z[2])
        c = f * c + i * c_hat
        h = o * np.tanh(c)
        if show:
            print(f"  t={t} {words[idx]:5} z={np.round(z, 3).tolist()} "
                  f"i={i:.3f} f={f:.3f} c̃={c_hat:.3f} o={o:.3f} → c={c:.3f} h={h:.3f}")
    return h

# Прямой проход
H = []
for s, seq in zip(sentences, train_pad):
    print(f'"{s}" {seq.tolist()}')
    H.append(lstm(seq, show=True))
H = np.array(H)

# model.compile(optimizer="rmsprop", loss="binary_crossentropy"); fit(batch_size=2)
lr, rho, eps = 0.001, 0.9, 1e-7
v_w = v_b = 0.0
for step in range(1, 4):
    p = sigmoid(w_d * H + b_d)
    loss = -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))
    g_w = np.mean((p - labels) * H)     # ∂L/∂w для пары sigmoid + BCE
    g_b = np.mean(p - labels)
    v_w = rho * v_w + (1 - rho) * g_w ** 2
    v_b = rho * v_b + (1 - rho) * g_b ** 2
    w_d -= lr * g_w / (np.sqrt(v_w) + eps)
    b_d -= lr * g_b / (np.sqrt(v_b) + eps)
    print(f"шаг {step}: p={np.round(p, 4).tolist()} loss={loss:.4f} "
          f"→ w={w_d:.4f} b={b_d:.4f}")

# Сокращённый оригинал: код Packt выше, но на двух отзывах и с
# круглыми весами — теми же, что в разборе на числах ниже.
# Блоки 1–6 пронумерованы так же, как там.
# «было:» — так в оригинале; «добавлено:» — нужно для сверки с разбором.
# Запуск: pip install tensorflow (в нём Keras 3), затем python этот_файл.py

import numpy as np  # было: import tensorflow as tf; import pandas as pd
from keras import preprocessing
from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential

# ── 1. Данные ──────────────────────────────────────────────
# было: df = pd.read_csv("SST-2/train.tsv", sep="\t")
#       sentences = df.sentence; labels = df.label
sentences = ["very good movie", "bad movie"]
labels = np.array([1, 0])

# ── 2. Словарь: слово → номер ──────────────────────────────
max_sen_len = max([len(s.split()) for s in sentences])
# было: list(set([...])) — порядок слов от запуска к запуску разный;
# dict.fromkeys сохраняет порядок появления — номера как в разборе
words = ["PAD"] + list(dict.fromkeys(
    [w for s in sentences for w in s.split()]))
word2idx = {w: i for i, w in enumerate(words)}
max_words = max(word2idx.values()) + 1
idx2word = {i: w for i, w in enumerate(words)}
print("word2idx:", word2idx)  # добавлено

# ── 3. Отзывы → номера, выравнивание ───────────────────────
train = [
    list(map(lambda x: word2idx[x], s.split()))
    for s in sentences
]
train_pad = preprocessing.sequence.pad_sequences(
    train,
    maxlen=max_sen_len,
)
print("Train shape:", train_pad.shape)
print("train_pad:", train_pad.tolist())  # добавлено

# ── 4. Модель: веса слоёв и что каждый слой считает ────────
model = Sequential()
model.add(Embedding(max_words, 2))  # было: 32 числа на слово
model.add(LSTM(1))                  # было: 32 ячейки
model.add(Dense(1, activation="sigmoid"))

# добавлено: вместо случайных начальных весов — круглые, как в разборе
model.build(input_shape=(None, max_sen_len))
model.layers[0].set_weights([np.array([   # E: строка на слово
    [0.0, 0.0], [0.2, 0.4], [0.9, 0.1], [0.1, 0.5], [-0.8, 0.3]])])
model.layers[1].set_weights([
    np.array([[0.5, 0.3, 1.0, 0.4],       # W от x[0]; столбцы i f c~ o
              [0.2, -0.1, 0.3, 0.1]]),    # W от x[1]
    np.array([[0.3, 0.2, 0.5, 0.2]]),     # U: от h прошлого слова
    np.array([0.0, 1.0, 0.0, 0.0]),       # b
])
model.layers[2].set_weights([np.array([[2.0]]), np.array([0.0])])  # w, b
# добавлено: Embedding и LSTM не обучаются — в разборе учится только Dense
model.layers[0].trainable = False
model.layers[1].trainable = False

# ── 5. compile: чем мерить ошибку, как менять веса ─────────
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["acc"],
)

# ── 6. fit: обучение ───────────────────────────────────────
history = model.fit(
    train_pad,
    labels,
    epochs=3,       # было: 30
    batch_size=32,  # как в книге: 2 отзыва — один батч на эпоху
    # было: validation_split=0.2 — из двух отзывов 20 % не отделить
)

# было: plot_graphs(history, "acc"); plot_graphs(history, "loss")
# добавлено: веса Dense после обучения — сверить с разбором
w, b = model.layers[2].get_weights()
print(f"Dense после обучения: w={w[0][0]:.4f} b={b[0]:.4f}")

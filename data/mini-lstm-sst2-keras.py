# Сокращённый оригинал: код Packt выше, но на двух отзывах и в размере,
# который можно пересчитать руками. Веса — настоящие стартовые веса Keras
# (seed закреплён, чтобы при каждом запуске выходили те же числа),
# учатся все слои, как в оригинале. Блоки 1–6 пронумерованы так же,
# как в разборе ниже. «было:» — так в оригинале; «добавлено:» — нужно
# для сверки с разбором.
# Запуск: pip install keras torch (Keras 3 на бэкенде torch), затем
# python этот_файл.py

import os
os.environ.setdefault("KERAS_BACKEND", "torch")  # добавлено: бэкенд, на котором снят вывод
import numpy as np  # было: import tensorflow as tf; import pandas as pd
import keras
from keras import preprocessing
from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential

keras.utils.set_random_seed(0)  # добавлено: стартовые веса случайные, но одни и те же


def show(name, a):  # добавлено: печать весов с 6 значащими цифрами; их копирует разбор
    print(f"{name} =", np.vectorize(lambda v: float(f"{v:.6g}") + 0.0)(a).tolist())


def show4(name, a):  # добавлено: 4 знака после запятой — сверка с разбором после обучения
    print(f"{name} =", (np.round(np.asarray(a, dtype=float), 4) + 0.0).tolist())


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

# добавлено: создать веса сейчас и напечатать стартовые — с них начинает разбор
model.build(input_shape=(None, max_sen_len))
E = model.layers[0].get_weights()[0]           # Embedding: строка на слово
W, U, b = model.layers[1].get_weights()        # LSTM: от x, от h, смещения; столбцы i f c~ o
w_d, b_d = model.layers[2].get_weights()       # Dense: вес и смещение
print("стартовые веса (seed 0):")
show("E", E)
show("W", W)
show("U", U[0])
show("b", b)
show("w_d", w_d[0][0])
show("b_d", b_d[0])

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
# добавлено: веса после обучения (4 знака) — сверить с разбором
E = model.layers[0].get_weights()[0]
W, U, b = model.layers[1].get_weights()
w_d, b_d = model.layers[2].get_weights()
print("после обучения:")
show4("E", E)
show4("W", W)
show4("U", U[0])
show4("b", b)
show4("w_d", w_d[0][0])
show4("b_d", b_d[0])

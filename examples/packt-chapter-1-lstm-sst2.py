# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH01/CH01_Introduction.ipynb
# Сначала выполнить в терминале или отдельной ячейке Jupyter:
# wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
# unzip SST-2.zip

import tensorflow as tf
import pandas as pd

df = pd.read_csv("SST-2/train.tsv", sep="\t")
sentences = df.sentence
labels = df.label

max_sen_len = max([len(s.split()) for s in sentences])
words = ["PAD"] + list(set([w for s in sentences for w in s.split()]))
word2idx = {w: i for i, w in enumerate(words)}
max_words = max(word2idx.values()) + 1
idx2word = {i: w for i, w in enumerate(words)}
train = [
    list(map(lambda x: word2idx[x], s.split()))
    for s in sentences
]

from keras import preprocessing
train_pad = preprocessing.sequence.pad_sequences(
    train,
    maxlen=max_sen_len,
)
print("Train shape:", train_pad.shape)

from keras.layers import LSTM, Embedding, Dense
from keras.models import Sequential

model = Sequential()
model.add(Embedding(max_words, 32))
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["acc"],
)
history = model.fit(
    train_pad,
    labels,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
)

import matplotlib.pyplot as plt

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_" + metric])
    plt.show()

plot_graphs(history, "acc")
plot_graphs(history, "loss")

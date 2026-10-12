# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH07/CH07a_Benchmarking_sentence_similarity_models.ipynb
import math
import tensorflow as tf
from sentence_transformers import SentenceTransformer

distilroberta = SentenceTransformer("stsb-distilroberta-base-v2")

def roberta_sts_benchmark(batch):
    vector_1 = tf.nn.l2_normalize(
        distilroberta.encode(batch["sentence1"]), axis=1
    )
    vector_2 = tf.nn.l2_normalize(
        distilroberta.encode(batch["sentence2"]), axis=1
    )
    cosine = tf.reduce_sum(tf.multiply(vector_1, vector_2), axis=1)
    cosine = tf.clip_by_value(cosine, -1.0, 1.0)
    return 1.0 - tf.acos(cosine) / math.pi

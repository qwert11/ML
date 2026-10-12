# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH01/CH01_Introduction.ipynb
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

toy_corpus = ["the fat cat sat on the mat",
              "the big cat slept",
              "the dog chased a cat"]
vectorizer = TfidfVectorizer(use_idf=True)
corpus_tfidf = vectorizer.fit_transform(toy_corpus)

print(f"The vocabulary size is {len(vectorizer.vocabulary_.keys())}")
print(f"The document-term matrix shape is {corpus_tfidf.shape}")

df = pd.DataFrame(np.round(corpus_tfidf.toarray(), 2))
df.columns = vectorizer.get_feature_names()
print(df)

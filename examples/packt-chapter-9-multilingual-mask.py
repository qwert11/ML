# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH09/CH09a_XLM_and_multilingual_BERT.ipynb
from transformers import pipeline

unmasker = pipeline(
    "fill-mask",
    model="bert-base-multilingual-uncased"
)
sentences = [
    "Transformers changed the [MASK] language processing",
    "Transformerlar [MASK] dil işlemeyi değiştirdiler",
    "ترنسفرمرها پردازش زبان [MASK] را تغییر دادند",
]

for sentence in sentences:
    print(sentence)
    print(unmasker(sentence)[0]["sequence"])
    print("=" * 50)

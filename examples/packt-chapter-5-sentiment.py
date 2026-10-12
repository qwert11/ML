# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH05/CH05a_BERT_fine-tuning.ipynb
from transformers import (
    pipeline,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

model = DistilBertForSequenceClassification.from_pretrained("MyBestIMDBModel")
tokenizer = DistilBertTokenizerFast.from_pretrained("MyBestIMDBModel")
sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(sentiment("the movie was very impressive"))
print(sentiment("the script of the picture was very poor"))

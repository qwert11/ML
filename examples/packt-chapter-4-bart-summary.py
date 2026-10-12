# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH04/CH04a_Summarization_with_BART.ipynb
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline

model_name = "sshleifer/distilbart-cnn-12-6"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
text = "The product looks good, but its stones fall out and it cannot be sterilized safely."
result = summarizer(text)
print(result[0]["summary_text"])

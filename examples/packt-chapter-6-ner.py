# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH06/CH06a_Fine_tuning_language_models_for_NER.ipynb
from transformers import AutoModelForTokenClassification, pipeline

mmodel = AutoModelForTokenClassification.from_pretrained("ner_model")
nlp = pipeline("ner", model=mmodel, tokenizer=tokenizer)

example = "I live in Istanbul"
ner_results = nlp(example)
print(ner_results)

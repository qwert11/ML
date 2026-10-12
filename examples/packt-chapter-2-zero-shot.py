# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH02/CH02a_Working_with_Language_Models_and_Tokenizers.ipynb
from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
sequence_to_classify = "I am going to france."
candidate_labels = ["travel", "cooking", "dancing"]
print(classifier(sequence_to_classify, candidate_labels))

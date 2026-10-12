# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH11/CH11a_Visualization_with_BertViz.ipynb
from bertviz import head_view
from transformers import BertTokenizer, BertModel

def get_bert_attentions(model_path, sentence_a, sentence_b):
    model = BertModel.from_pretrained(
        model_path,
        output_attentions=True,
    )
    tokenizer = BertTokenizer.from_pretrained(model_path)
    inputs = tokenizer.encode_plus(
        sentence_a,
        sentence_b,
        return_tensors="pt",
        add_special_tokens=True,
    )
    token_type_ids = inputs["token_type_ids"]
    input_ids = inputs["input_ids"]
    attention = model(input_ids, token_type_ids=token_type_ids)[-1]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    return attention, tokens

model_path = "bert-base-cased"
sentence_a = "The cat is very sad."
sentence_b = "Because it could not find food to eat."
attention, tokens = get_bert_attentions(model_path, sentence_a, sentence_b)
head_view(attention, tokens)

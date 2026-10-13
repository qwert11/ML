// Короткие фрагменты из официального репозитория Packt.
// Лицензия исходного кода: MIT, Copyright (c) 2021 Packt.
const commit = 'f088d75fb24f583a581a5a99fa02ccf445b77dfc';
const repository = 'https://github.com/PacktPublishing/Mastering-Transformers';
const sourceUrl = path => `${repository}/blob/${commit}/${path}`;

module.exports = [
  {
    id: 'chapter-1-lstm-sst2', bookId: 'yildirim-mastering-transformers', chapter: 1, page: '32–33',
    title: 'Первый пример книги: анализ тональности SST-2 с LSTM',
    purpose: 'Загружаем размеченные отзывы SST-2, строим словарь, выравниваем последовательности и обучаем LSTM отличать положительную тональность от отрицательной.',
    sourcePath: 'CH01/CH01_Introduction.ipynb', cells: 'ячейки 40–51',
    sourceUrl: sourceUrl('CH01/CH01_Introduction.ipynb'), repository,
    requires: ['TensorFlow / Keras', 'pandas', 'matplotlib', 'SST-2 из GLUE'],
    compatibility: 'Код книги использует отдельный пакет keras и простое разбиение строк по пробелам. В современном TensorFlow обычно импортируют tensorflow.keras; для реального проекта нужен токенизатор с обработкой неизвестных слов.',
    excerptNote: 'Сверено по фотографиям русского издания: стр. 32–33. В печатной книге model.fit использует 30 эпох и batch_size=32; в текущем официальном notebook — 15 эпох и batch_size=16. Ниже сохранены параметры печатной книги.',
    relatedFormulaIds: ['sigmoid', 'tanh-activation', 'mlp-cost'],
    miniDemo: require('./mini-lstm-sst2.cjs'),
    code: `# Сначала выполнить в терминале или отдельной ячейке Jupyter:
# wget https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
# unzip SST-2.zip

import tensorflow as tf
import pandas as pd

df = pd.read_csv("SST-2/train.tsv", sep="\\t")
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
plot_graphs(history, "loss")`,
    walkthrough: [
      'SST-2 загружается из набора GLUE; train.tsv содержит текст предложения и бинарную метку тональности.',
      'Каждое уникальное слово получает целочисленный индекс word2idx, после чего предложения превращаются из строк в списки индексов.',
      'pad_sequences дополняет короткие предложения значением PAD и обрезает длинные до max_sen_len; для этого набора книга показывает форму (67349, 52).',
      'Embedding обучает 32-мерный вектор каждого слова, а LSTM(32) сворачивает последовательность в состояние из 32 признаков.',
      'Dense с сигмоидой выдаёт вероятность положительного класса; binary_crossentropy штрафует неверную вероятность.',
      'validation_split=0.2 откладывает пятую часть примеров, а plot_graphs сравнивает обучение и валидацию по accuracy и loss.'
    ]
  },
  {
    id: 'chapter-2-zero-shot', bookId: 'yildirim-mastering-transformers', chapter: 2,
    title: 'Zero-shot классификация с BART',
    purpose: 'Классифицируем фразу по произвольным меткам без отдельного обучения на этих классах.',
    sourcePath: 'CH02/CH02a_Working_with_Language_Models_and_Tokenizers.ipynb', cells: 'ячейка 8',
    sourceUrl: sourceUrl('CH02/CH02a_Working_with_Language_Models_and_Tokenizers.ipynb'), repository,
    requires: ['transformers', 'PyTorch'],
    compatibility: 'Первый запуск скачивает facebook/bart-large-mnli и требует заметный объём памяти.',
    excerptNote: 'Официальная ячейка приведена целиком.',
    relatedFormulaIds: ['softmax'],
    code: `from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)
sequence_to_classify = "I am going to france."
candidate_labels = ["travel", "cooking", "dancing"]
print(classifier(sequence_to_classify, candidate_labels))`,
    walkthrough: [
      'pipeline загружает токенизатор и BART, обученный на задаче natural language inference.',
      'Исходная фраза по очереди сопоставляется с гипотезами, построенными из названий классов.',
      'Модель возвращает оценки классов, а softmax превращает их в нормированные вероятности.',
      'Новые подписи можно менять без повторного обучения модели.'
    ]
  },
  {
    id: 'chapter-3-train-bert', bookId: 'yildirim-mastering-transformers', chapter: 3,
    title: 'Обучение маленького BERT для masked language modeling',
    purpose: 'Создаём WordPiece-токенизатор, набор маскированных примеров и запускаем обучение BERT с нуля.',
    sourcePath: 'CH03/CH03a_BERT_As_one_of_Autoencoding_Language_Models.ipynb', cells: 'ячейки 8, 10, 16–22',
    sourceUrl: sourceUrl('CH03/CH03a_BERT_As_one_of_Autoencoding_Language_Models.ipynb'), repository,
    requires: ['tokenizers', 'transformers', 'PyTorch', 'corpus.txt'],
    compatibility: 'LineByLineTextDataset относится к старому API Transformers. Страница сохраняет оригинальный подход книги и отдельно отмечает это ограничение.',
    excerptNote: 'Последовательные ячейки официального ноутбука собраны в один блок; промежуточный просмотр словаря опущен.',
    relatedFormulaIds: ['masked-lm-loss', 'multi-head-attention', 'layer-normalization'],
    code: `from tokenizers import BertWordPieceTokenizer

wordpiece = BertWordPieceTokenizer()
wordpiece.train("corpus.txt")
wordpiece.save_model("tokenizer")

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("tokenizer")

from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="corpus.txt",
    block_size=128
)

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

from transformers import TrainingArguments, BertConfig, BertForMaskedLM, Trainer
training_args = TrainingArguments(
    output_dir="BERT",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=128
)
bert = BertForMaskedLM(BertConfig())
trainer = Trainer(
    model=bert,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)
trainer.train()`,
    walkthrough: [
      'WordPiece изучает части слов непосредственно по локальному corpus.txt.',
      'LineByLineTextDataset режет корпус на последовательности не длиннее 128 токенов.',
      'DataCollator случайно скрывает 15% токенов и формирует правильные ответы только для скрытых позиций.',
      'BertForMaskedLM создаётся из конфигурации без предобученных весов.',
      'Trainer объединяет модель, данные и параметры обучения; функция потерь соответствует masked LM.'
    ]
  },
  {
    id: 'chapter-4-bart-summary', bookId: 'yildirim-mastering-transformers', chapter: 4,
    title: 'Абстрактивная суммаризация с DistilBART',
    purpose: 'Загружаем готовую encoder-decoder модель и получаем краткое содержание пользовательского отзыва.',
    sourcePath: 'CH04/CH04a_Summarization_with_BART.ipynb', cells: 'ячейки 1, 4–5',
    sourceUrl: sourceUrl('CH04/CH04a_Summarization_with_BART.ipynb'), repository,
    requires: ['transformers', 'PyTorch'],
    compatibility: 'Модель скачивается с Hugging Face. Для длинных текстов нужно учитывать ограничение длины токенов.',
    excerptNote: 'Инициализация модели сохранена; длинный отзыв из ноутбука сокращён, чтобы пример удобно читался на странице.',
    relatedFormulaIds: ['causal-lm', 'multi-head-attention'],
    code: `from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import pipeline

model_name = "sshleifer/distilbart-cnn-12-6"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
text = "The product looks good, but its stones fall out and it cannot be sterilized safely."
result = summarizer(text)
print(result[0]["summary_text"])`,
    walkthrough: [
      'BART использует encoder для представления всего исходного текста.',
      'Decoder генерирует резюме слева направо, учитывая уже созданные токены.',
      'DistilBART — уменьшенная версия модели, настроенная на суммаризацию CNN/DailyMail.',
      'pipeline скрывает токенизацию, вызов generate и обратное преобразование токенов в текст.'
    ]
  },
  {
    id: 'chapter-5-sentiment', bookId: 'yildirim-mastering-transformers', chapter: 5,
    title: 'Проверка дообученного DistilBERT на тональности',
    purpose: 'Загружаем сохранённую после fine-tuning модель и классифицируем положительную и отрицательную фразы.',
    sourcePath: 'CH05/CH05a_BERT_fine-tuning.ipynb', cells: 'ячейки 24–25, 28–29',
    sourceUrl: sourceUrl('CH05/CH05a_BERT_fine-tuning.ipynb'), repository,
    requires: ['transformers', 'PyTorch', 'каталог MyBestIMDBModel из предыдущих ячеек'],
    compatibility: 'Код предполагает, что модель уже дообучена и сохранена предыдущими ячейками ноутбука.',
    excerptNote: 'Показан официальный блок инференса после обучения; длинная конфигурация Trainer остаётся в исходном ноутбуке.',
    relatedFormulaIds: ['softmax'],
    code: `from transformers import (
    pipeline,
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
)

model = DistilBertForSequenceClassification.from_pretrained("MyBestIMDBModel")
tokenizer = DistilBertTokenizerFast.from_pretrained("MyBestIMDBModel")
sentiment = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(sentiment("the movie was very impressive"))
print(sentiment("the script of the picture was very poor"))`,
    walkthrough: [
      'from_pretrained читает локальные веса, конфигурацию классов и словарь токенизатора.',
      'pipeline превращает текст в input_ids и attention_mask.',
      'Классификационная голова получает представление последовательности и выдаёт два логита.',
      'Softmax преобразует логиты в вероятность положительного или отрицательного класса.'
    ]
  },
  {
    id: 'chapter-6-ner', bookId: 'yildirim-mastering-transformers', chapter: 6,
    title: 'Распознавание именованных сущностей',
    purpose: 'Находим в предложении сущность Istanbul с помощью дообученной модели token classification.',
    sourcePath: 'CH06/CH06a_Fine_tuning_language_models_for_NER.ipynb', cells: 'ячейки 26–28',
    sourceUrl: sourceUrl('CH06/CH06a_Fine_tuning_language_models_for_NER.ipynb'), repository,
    requires: ['transformers', 'PyTorch', 'обученная ner_model и tokenizer'],
    compatibility: 'Перед этим блоком ноутбук обучает модель на CoNLL-2003 и записывает id2label/label2id в config.json.',
    excerptNote: 'Официальный финальный блок инференса приведён без этапа обучения.',
    relatedFormulaIds: ['softmax'],
    code: `from transformers import AutoModelForTokenClassification, pipeline

mmodel = AutoModelForTokenClassification.from_pretrained("ner_model")
nlp = pipeline("ner", model=mmodel, tokenizer=tokenizer)

example = "I live in Istanbul"
ner_results = nlp(example)
print(ner_results)`,
    walkthrough: [
      'Токенизатор разбивает предложение и сохраняет соответствие токенов исходным словам.',
      'Модель выдаёт отдельный набор логитов классов NER для каждой позиции.',
      'Для Istanbul выбирается метка географической сущности, обученная на CoNLL-2003.',
      'pipeline возвращает метку, уверенность, токен и его позицию в строке.'
    ]
  },
  {
    id: 'chapter-7-sentence-similarity', bookId: 'yildirim-mastering-transformers', chapter: 7,
    title: 'Сравнение предложений через косинусное сходство',
    purpose: 'Кодируем пары предложений Sentence-BERT, нормализуем векторы и сравниваем направления.',
    sourcePath: 'CH07/CH07a_Benchmarking_sentence_similarity_models.ipynb', cells: 'ячейки 4, 6–10',
    sourceUrl: sourceUrl('CH07/CH07a_Benchmarking_sentence_similarity_models.ipynb'), repository,
    requires: ['sentence-transformers', 'TensorFlow', 'datasets', 'набор STS-B'],
    compatibility: 'Название модели и API load_metric относятся к версии библиотек времени выхода книги.',
    excerptNote: 'Сохранена официальная функция оценки DistilRoBERTa; загрузка STS-B и второй сравниваемой модели опущены.',
    relatedFormulaIds: ['cosine-similarity'],
    code: `import math
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
    return 1.0 - tf.acos(cosine) / math.pi`,
    walkthrough: [
      'SentenceTransformer преобразует каждое предложение в один плотный вектор.',
      'L2-нормализация делает длину каждого вектора равной единице.',
      'После нормализации скалярное произведение равно косинусному сходству.',
      'arccos переводит сходство в угловую меру, затем результат масштабируется к удобному диапазону.'
    ]
  },
  {
    id: 'chapter-8-pruning-quantization', bookId: 'yildirim-mastering-transformers', chapter: 8,
    title: 'Pruning и динамическая квантизация',
    purpose: 'Удаляем 20% малых весов и переводим линейные слои Sentence-BERT в 8-битное представление.',
    sourcePath: 'CH08/CH08a_Pruning_Quantization.ipynb', cells: 'ячейки 2, 7–8, 13',
    sourceUrl: sourceUrl('CH08/CH08a_Pruning_Quantization.ipynb'), repository,
    requires: ['sentence-transformers', 'PyTorch'],
    compatibility: 'Pruning API в новых версиях PyTorch отличается. Этот блок сохранён как оригинальный пример книги, а не как обещание запуска на любой современной версии.',
    excerptNote: 'Официальные ячейки загрузки, pruning и quantization объединены; оценка STS-B показана в исходном ноутбуке.',
    relatedFormulaIds: ['l1-regularization', 'cosine-similarity'],
    code: `import torch
from torch.nn.utils import prune
from sentence_transformers import SentenceTransformer

distilroberta = SentenceTransformer("stsb-distilroberta-base-v2")
pruner = prune.L1Unstructured(amount=0.2)

state_dict = distilroberta.state_dict()
for key in state_dict.keys():
    if "weight" in key:
        state_dict[key] = pruner.prune(state_dict[key])
distilroberta.load_state_dict(state_dict)

distilroberta = torch.quantization.quantize_dynamic(
    model=distilroberta,
    qconfig_spec={torch.nn.Linear: torch.quantization.default_dynamic_qconfig},
    dtype=torch.qint8,
)`,
    walkthrough: [
      'L1Unstructured ранжирует веса по абсолютной величине и зануляет долю amount=0.2.',
      'Цикл применяет разреживание только к параметрам, в имени которых есть weight.',
      'Динамическая квантизация заменяет вычисления Linear на вариант с int8-весами.',
      'После каждого шага в ноутбуке повторно измеряется качество на STS-B, чтобы увидеть цену сжатия.'
    ]
  },
  {
    id: 'chapter-9-multilingual-mask', bookId: 'yildirim-mastering-transformers', chapter: 9,
    title: 'Multilingual BERT заполняет маску на трёх языках',
    purpose: 'Проверяем одну multilingual-модель на английской, турецкой и персидской фразах.',
    sourcePath: 'CH09/CH09a_XLM_and_multilingual_BERT.ipynb', cells: 'ячейки 2–3',
    sourceUrl: sourceUrl('CH09/CH09a_XLM_and_multilingual_BERT.ipynb'), repository,
    requires: ['transformers', 'PyTorch'],
    compatibility: 'Первый запуск скачивает bert-base-multilingual-uncased.',
    excerptNote: 'Две официальные ячейки объединены без изменения входных предложений.',
    relatedFormulaIds: ['masked-lm-loss'],
    code: `from transformers import pipeline

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
    print("=" * 50)`,
    walkthrough: [
      'Один общий словарь токенизирует тексты с разными письменностями.',
      'BERT видит контекст слева и справа от [MASK].',
      'Голова masked LM выдаёт логиты по всему словарю для скрытой позиции.',
      'В примере печатается самый вероятный вариант заполнения для каждого языка.'
    ]
  },
  {
    id: 'chapter-10-fastapi-qa', bookId: 'yildirim-mastering-transformers', chapter: 10,
    title: 'Публикация question-answering модели через FastAPI',
    purpose: 'Оборачиваем готовую модель вопрос-ответ в HTTP POST endpoint.',
    sourcePath: 'CH10/CH10a_fastAPI_Transformer_model_serving/main.py', cells: 'полный файл main.py',
    sourceUrl: sourceUrl('CH10/CH10a_fastAPI_Transformer_model_serving/main.py'), repository,
    requires: ['fastapi', 'uvicorn', 'pydantic', 'transformers', 'PyTorch'],
    compatibility: 'Сервер скачивает distilbert-base-cased-distilled-squad при первом запуске. Запускайте только в доверенной среде.',
    excerptNote: 'Это полный официальный Python-файл главы 10; изменено только форматирование пробелов.',
    relatedFormulaIds: ['softmax'],
    code: `import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

class QADataModel(BaseModel):
    question: str
    context: str

app = FastAPI()
model_name = "distilbert-base-cased-distilled-squad"
model = pipeline(
    model=model_name,
    tokenizer=model_name,
    task="question-answering",
)

@app.post("/question_answering")
async def qa(input_data: QADataModel):
    result = model(
        question=input_data.question,
        context=input_data.context,
    )
    return {"answer": result["answer"]}

if __name__ == "__main__":
    uvicorn.run("main:app", workers=1)`,
    walkthrough: [
      'QADataModel задаёт контракт запроса: вопрос и контекст должны быть строками.',
      'pipeline загружает DistilBERT, дообученный искать начало и конец ответа.',
      'POST endpoint передаёт обе строки модели и получает найденный фрагмент.',
      'FastAPI сериализует ответ в JSON, а Uvicorn обслуживает приложение.'
    ]
  },
  {
    id: 'chapter-11-bertviz', bookId: 'yildirim-mastering-transformers', chapter: 11,
    title: 'Визуализация голов внимания с BertViz',
    purpose: 'Извлекаем attention-тензоры BERT и интерактивно показываем связи между токенами.',
    sourcePath: 'CH11/CH11a_Visualization_with_BertViz.ipynb', cells: 'ячейки 2–5',
    sourceUrl: sourceUrl('CH11/CH11a_Visualization_with_BertViz.ipynb'), repository,
    requires: ['bertviz', 'transformers', 'PyTorch', 'Jupyter Notebook'],
    compatibility: 'head_view выводит интерактивный виджет и предназначен для Jupyter/Colab.',
    excerptNote: 'Официальная функция и первый вызов объединены в один блок.',
    relatedFormulaIds: ['scaled-dot-product-attention', 'multi-head-attention'],
    code: `from bertviz import head_view
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
head_view(attention, tokens)`,
    walkthrough: [
      'output_attentions=True просит BERT вернуть матрицы внимания всех слоёв и голов.',
      'encode_plus объединяет два предложения и добавляет служебные токены.',
      'Модель возвращает веса внимания вместе с обычными скрытыми состояниями.',
      'BertViz сопоставляет строки и столбцы матриц токенам и рисует связи для выбранной головы.'
    ]
  }
];

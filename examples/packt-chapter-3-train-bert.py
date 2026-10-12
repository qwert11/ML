# Фрагмент официального кода Packt: Mastering Transformers
# MIT License, Copyright (c) 2021 Packt
# Источник: https://github.com/PacktPublishing/Mastering-Transformers/blob/f088d75fb24f583a581a5a99fa02ccf445b77dfc/CH03/CH03a_BERT_As_one_of_Autoencoding_Language_Models.ipynb
from tokenizers import BertWordPieceTokenizer

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
trainer.train()

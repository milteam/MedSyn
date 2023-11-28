import json
import pandas as pd
import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np

#model_name = "xlm-roberta-base" #dont work№ model_name = "DeepPavlov/xlm-roberta-large-en-ru"

BATCH_SIZE = 64


max_input_length = 256
max_sentence_length = 128


def trim_sentence(sentence):
  # splitting the sentence
  sentence = sentence.split()
  # check if the sentence has 128 or more tokens
  if len(sentence) >= 128:
    sentence = sentence[:max_sentence_length]
  return " ".join(sentence)


LABELS = {"neutral": 0, "entailment": 1, "contradiction": 2}

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

# In[ ]:
def create_csv(source, target):
    df = []
    with open(source) as f:
        for row in f.readlines():
            df.append(json.loads(row))

    df = pd.DataFrame(df)
    df['sentence1'] = df['ru_sentence1'].apply(trim_sentence)
    df['sentence2'] = df['ru_sentence2'].apply(trim_sentence)

    # df['t_sentence1'] = cls_token + ' ' + df['sentence1'] + ' ' + sep_token + ' '
    # df['t_sentence2'] = cls_token + ' ' + df['sentence2'] + ' ' + sep_token + ' '
    df['sequence'] = df['sentence1'] + ' относится к ' + df['sentence2']
    df['gold_label'] = df['gold_label'].apply(lambda x: LABELS[x])


    df = df[['gold_label', 'sequence']]
    #df = df.head(32)
    df.to_csv(target, index=False)
    return df

def train_huggingface(train, val, pred, checkpoint, bert, epochs):

    create_csv(train, "train.csv")
    test_df = create_csv(val, "dev.csv")

    # In[ ]:
    dataset = load_dataset('csv', data_files={'train': "train.csv",
                                            'test': 'dev.csv'})

    model = AutoModelForSequenceClassification.from_pretrained(bert, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(bert)

    def encode(examples):
        return tokenizer(examples['sequence'], truncation=True, padding=True, return_tensors="pt")


    dataset = dataset.map(encode, batched=True)


    dataset = dataset.map(lambda examples: {'label': examples['gold_label']}, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'sequence', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(checkpoint,
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",
                                      save_total_limit=2,
                                      num_train_epochs=epochs,
                                      per_device_train_batch_size=BATCH_SIZE,
                                      per_device_eval_batch_size=BATCH_SIZE,
                                      # warmup_steps=500,
                                      # weight_decay=0.01,
                                      logging_dir="bert_results/logs",
                                      logging_strategy="epoch",
                                      load_best_model_at_end=True,
                                      report_to="none"
                                      )





    trainer = Trainer(
        model,
        training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics

    )

    # model.eval()
    # predictions = trainer.predict(dataset["test"])
    # print(predictions)
    # predictions = np.argmax(predictions.predictions, axis=-1)
    # test_df["predictions"] = predictions
    # test_df.to_csv("predictions.csv", index=False)

    trainer.train()
    model.eval()
    predictions = trainer.predict(dataset["test"])
    print(predictions)
    predictions = np.argmax(predictions.predictions, axis=-1)
    test_df["predictions"] = predictions
    test_df.to_csv(pred, index=False)
    trainer.save_model(checkpoint)

def predict_huggingface(val, pred, checkpoint):

    test_df = create_csv(val, "dev.csv")

    # In[ ]:
    dataset = load_dataset('csv', data_files={
                                            'test': 'dev.csv'})

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def encode(examples):
        return tokenizer(examples['sequence'], truncation=True, padding=True, return_tensors="pt")


    dataset = dataset.map(encode, batched=True)


    dataset = dataset.map(lambda examples: {'label': examples['gold_label']}, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'sequence', 'label'])

    model.eval()
    predictions = model.predict(dataset["test"])
    print(predictions)
    predictions = np.argmax(predictions.predictions, axis=-1)
    test_df["predictions"] = predictions
    test_df.to_csv(pred, index=False)

import json
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer, AutoModel, T5EncoderModel
import numpy as np

#model_name = "xlm-roberta-base" #dont work№ model_name = "DeepPavlov/xlm-roberta-large-en-ru"

BATCH_SIZE = 16


max_input_length = 512

class T5SequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(T5SequenceClassification, self).__init__()
        if "T5" in pretrained_model_name:
            MODEL = T5EncoderModel
        else:
            MODEL = AutoModel
        self.t5_model = MODEL.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.t5_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.t5_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


LABELS = {"neutral": 0, "entailment": 1, "contradiction": 2}

top_k = 3

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argsort(-predictions)[:,0:top_k]
    acc_at_k = sum([l in p for l, p in zip(labels, preds)])/len(labels)
    return {'acc_at_k': acc_at_k}

# In[ ]:
def create_csv(source, target, labels=None):
    df = []
    with open(source) as f:
        for row in f.readlines():
            df.append(json.loads(row))

    df = pd.DataFrame(df)

    if not labels:
        labels = {label: idx for idx, label in enumerate(df["code"].unique())}

    df['code'] = df['code'].apply(lambda x: labels[x])
    df["sequence"] = df["symptoms"]

    df = df[['code', 'sequence']]
    #df = df.head(32)
    df.to_csv(target, index=False)
    return df, labels

def train_huggingface(train, val, pred, checkpoint, bert, epochs):

    _, labels = create_csv(train, "train.csv")
    test_df = create_csv(val, "dev.csv", labels)

    # In[ ]:
    dataset = load_dataset('csv', data_files={'train': "train.csv",
                                            'test': 'dev.csv'})

    model = AutoModelForSequenceClassification.from_pretrained(bert, num_labels=len(labels))
    tokenizer = AutoTokenizer.from_pretrained(bert)

    def encode(examples):
        return tokenizer(examples['sequence'], truncation=True, padding=True, return_tensors="pt", max_length=max_input_length)


    dataset = dataset.map(encode, batched=True)


    dataset = dataset.map(lambda examples: {'label': examples['code']}, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'sequence', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments("test-trainer",
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",
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
    test_df.to_csv("predictions.csv", index=False)

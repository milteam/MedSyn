import json
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np

# model_name = "xlm-roberta-base" #dont work
model_name = "DeepPavlov/xlm-roberta-large-en-ru"
#model_name = 'DeepPavlov/rubert-base-cased'

BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(model_name)

sep_token = tokenizer.sep_token #or tokenizer.eos_token
cls_token = tokenizer.cls_token #or ""
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(cls_token, sep_token, pad_token, unk_token) #2 0 1 3

sep_token_idx = tokenizer.sep_token_id
cls_token_idx = tokenizer.cls_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(sep_token_idx, cls_token_idx, pad_token_idx, unk_token_idx)

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
    df['sequence'] = df['sentence1'] + sep_token + df['sentence2']
    df['gold_label'] = df['gold_label'].apply(lambda x: LABELS[x])


    df = df[['gold_label', 'sequence']]
    df.to_csv(target, index=False)


create_csv("data/train_v1.jsonl", "train_v1.csv")
create_csv("data/dev_v1.jsonl", "dev_v1.csv")

# In[ ]:
dataset = load_dataset('csv', data_files={'train': "train_v1.csv",
                                        'test': 'dev_v1.csv'})

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def encode(examples):
    return tokenizer(examples['sequence'], truncation=True, padding='max_length')


dataset = dataset.map(encode, batched=True)


dataset = dataset.map(lambda examples: {'label': examples['gold_label']}, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'sequence', 'label'])

print(dataset['train'][:10])
print(dataset['test'][:10])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
training_args = TrainingArguments("test-trainer",
                                  evaluation_strategy="epoch",
                                  save_strategy="epoch",
                                  num_train_epochs=50,
                                  per_device_train_batch_size=BATCH_SIZE,
                                  per_device_eval_batch_size=BATCH_SIZE,
                                  warmup_steps=500,
                                  weight_decay=0.01,
                                  logging_dir="bert_results/logs",
                                  logging_strategy="epoch"
                                  )

metric = load_metric('accuracy')


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(predictions)
    print(labels)
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


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


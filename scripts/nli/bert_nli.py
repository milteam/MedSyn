import json
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer
import numpy as np

model_name = "xlm-roberta-base"
#model_name = 'DeepPavlov/rubert-base-cased'

BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained(model_name)

sep_token = tokenizer.sep_token
cls_token = tokenizer.cls_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(cls_token, sep_token, pad_token, unk_token)

sep_token_idx = tokenizer.sep_token_id
cls_token_idx = tokenizer.cls_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(sep_token_idx, cls_token_idx, pad_token_idx, unk_token_idx)

max_input_length = 256
max_sentence_length = 128

def tokenize_sentences(sentence):
  tokens = tokenizer.tokenize(sentence)
  return tokens

def reduce_sentence_length(sentence):
  tokens = sentence.strip().split(" ")
  tokens = tokens[:max_input_length]
  return tokens

def trim_sentence(sentence):
  # splitting the sentence
  sentence = sentence.split()
  # check if the sentence has 128 or more tokens
  if len(sentence) >= 128:
    sentence = sentence[:max_sentence_length]
  return " ".join(sentence)


def token_type_ids_sent_01(sentence):
  try:
    return [0] * len(sentence)
  except:
    return []


# In[ ]:


# function to get the token type id's of the sentence-02
def token_type_ids_sent_02(sentence):
  try:
    return [1] * len(sentence)
  except:
    return []


# Attention mask helps the model to know the useful tokens and padding that is done during batch preparation. Attention mask is basically a sequence of 1’s with the same length as input tokens.

# In[ ]:


# function to get the attention mask of the given sentence
def attention_mask_sentence(sentence):
  try:
    return [1] * len(sentence)
  except:
    return []


# In[ ]:


# function to combine the sequences from lists
def combine_sequence(sequence):
  return " ".join(sequence)

# function to combine the masks
def combine_mask(mask):
  mask = [str(m) for m in mask]
  return " ".join(mask)

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

    df['t_sentence1'] = cls_token + ' ' + df['sentence1'] + ' ' + sep_token + ' '
    df['t_sentence2'] = cls_token + ' ' + df['sentence2'] + ' ' + sep_token + ' '
    df['sequence'] = df['t_sentence1'] + df['t_sentence2']
    df['gold_label'] = df['gold_label'].apply(lambda x: LABELS[x])
    # df['b_sentence1'] = df['t_sentence1'].apply(tokenize_sentences)
    #df['t_sentence2'].apply(tokenize_sentences)
    #
    # df['sentence1_token_type'] = df['b_sentence1'].apply(token_type_ids_sent_01)
    # df['sentence2_token_type'] = df['b_sentence2'].apply(token_type_ids_sent_01)
    #
    # df['sequence'] = df['b_sentence1'] + df['b_sentence2']
    # df['attention_mask'] = df['sequence'].apply(attention_mask_sentence)
    # df['token_type'] = df['sentence1_token_type'] + df['sentence2_token_type']
    #
    # df['sequence'] = df['sequence'].apply(combine_sequence)
    # df['attention_mask'] = df['attention_mask'].apply(combine_mask)
    # df['token_type'] = df['token_type'].apply(combine_mask)

    df = df[['gold_label', 'sequence']]
    df.to_csv(target, index=False)


create_csv("data/train_v1.jsonl", "train_v1.csv")
create_csv("data/dev_v1.jsonl", "dev_v1.csv")

def convert_to_int(ids):
  ids = [int(d) for d in ids]
  return ids


# Create PyTorch Tensor using torchtext field

# In[ ]:

#
# # importing the saved data from csv file
# df_train = pd.read_csv('snli_1.0/snli_1.0_train.csv')
# df_dev = pd.read_csv('snli_1.0/snli_1.0_dev.csv')
# df_test = pd.read_csv('snli_1.0/snli_1.0_test.csv')



# In[ ]:
dataset = load_dataset('csv', data_files={'train': "train_v1.csv",
                                        'test': 'dev_v1.csv'})

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
#model.config.problem_type = "multi_label_classification"
def encode(examples):
    return tokenizer(examples['sequence'], truncation=True, padding='max_length')

dataset = dataset.map(encode, batched=True)

dataset = dataset.map(lambda examples: {'labels': examples['gold_label']}, batched=True)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
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


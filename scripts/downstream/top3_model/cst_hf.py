import json
import pandas as pd
import torch
from torch import nn
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, \
    Trainer, AutoModel, T5EncoderModel
import numpy as np

#model_name = "xlm-roberta-base" #dont work№ model_name = "DeepPavlov/xlm-roberta-large-en-ru"

BATCH_SIZE = 8


max_input_length = 512

class SequenceClassification(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(SequenceClassification, self).__init__()
        if "T5" in pretrained_model_name:
            MODEL = T5EncoderModel
        else:
            MODEL = AutoModel
        self.model = MODEL.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels, return_output=True, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        loss = self.loss(logits, labels)
        if return_output:
            return loss, logits
        else:
            return loss

# class CustomT5Trainer(Trainer):
#     def __init__(self, *args, **kwargs):
#         super(CustomT5Trainer, self).__init__(*args, **kwargs)
#         self.loss_fn = torch.nn.CrossEntropyLoss()
#
#     def compute_loss(self, model, inputs):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
#         logits = outputs
#         loss = self.loss_fn(logits, labels)
#         return loss
#
#     def compute_loss(self, model, inputs, return_outputs=False):
#         """
#         How the loss is computed by Trainer. By default, all models return the loss in the first element.
#
#         Subclass and override for custom behavior.
#         """
#         if self.label_smoother is not None and "labels" in inputs:
#             labels = inputs.pop("labels")
#         else:
#             labels = None
#         outputs = model(**inputs)
#         # Save past state if it exists
#         # TODO: this needs to be fixed and made cleaner later.
#         if self.args.past_index >= 0:
#             self._past = outputs[self.args.past_index]
#
#         if labels is not None:
#             if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
#                 loss = self.label_smoother(outputs, labels, shift_labels=True)
#             else:
#                 loss = self.label_smoother(outputs, labels)
#         else:
#             if isinstance(outputs, dict) and "loss" not in outputs:
#                 raise ValueError(
#                     "The model did not return a loss from the inputs, only the following keys: "
#                     f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
#                 )
#             # We don't use .loss here since the model may return tuples instead of ModelOutput.
#             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
#
#         return (loss, outputs) if return_outputs else loss

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

    model = SequenceClassification(bert, num_labels=len(labels))
    tokenizer = AutoTokenizer.from_pretrained(bert)

    def encode(examples):
        return tokenizer(examples['sequence'], truncation=True, padding=True, return_tensors="pt", max_length=max_input_length)


    dataset = dataset.map(encode, batched=True)


    dataset = dataset.map(lambda examples: {'label': examples['code']}, batched=True)
    dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'sequence', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments("test-trainer",
                                      evaluation_strategy="epoch",
                                      save_strategy="no",
                                      num_train_epochs=epochs,
                                      per_device_train_batch_size=BATCH_SIZE,
                                      per_device_eval_batch_size=BATCH_SIZE,
                                      warmup_steps=500,
                                      learning_rate=1e-03,
                                      weight_decay=0.01,
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

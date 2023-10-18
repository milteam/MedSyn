"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with softmax loss function. At every 1000 training steps, the model is evaluated on the
STS benchmark dataset

Usage:
python training_nli.py

OR
python training_nli.py pretrained_transformer_model_name
"""
import json

from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, LabelAccuracyEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Check if dataset exsist. If not, download and extract  it

#You can specify any huggingface/transformers pre-trained model here, for example, bert-base-uncased, roberta-base, xlm-roberta-base

# Read the dataset
train_batch_size = 16

def train_sentence_transformers(train, val, pred, checkpoint, bert, epochs):
    word_embedding_model = models.Transformer(bert)

    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}



    train_samples = []
    with open(train) as f:
        for row in f.readlines():
            row = json.loads(row)

            label_id = label2int[row['gold_label']]
            train_samples.append(InputExample(texts=[row['ru_sentence1'], row['ru_sentence2']], label=label_id))

    logging.info("Train samples: {}".format(len(train_samples)))


    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
    train_loss = losses.SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(), num_labels=len(label2int))

    dev_samples = []
    with open(val) as f:
        for row in f.readlines():
            row =json.loads(row)
            dev_samples.append(InputExample(texts=[row['ru_sentence1'], row['ru_sentence2']], label=label2int[row['gold_label']]))

    dev_dataloader = DataLoader(dev_samples, shuffle=True, batch_size=train_batch_size)

    #dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
    dev_evaluator = LabelAccuracyEvaluator(dev_dataloader, softmax_model=train_loss)
    # Configure the training
    num_epochs = 50

    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))



    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=dev_evaluator,
              epochs=num_epochs,
              evaluation_steps=1700,
              warmup_steps=warmup_steps,
              output_path=checkpoint
              )

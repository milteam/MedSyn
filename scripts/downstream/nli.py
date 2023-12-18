import click
import torch

from scripts.downstream.nli_model.hf import train_huggingface
from scripts.downstream.nli_model.sentence_bert import train_sentence_transformers


@click.command()
@click.option("--train", "-t", help="Train set jsonl path", required=True)
@click.option("--val", "-v", help="Validation set jsonl file", required=True)
@click.option("--pred", "-p", help="Predictions csv")
@click.option("--checkpoint", "-c", help="Path to store model", required=True)
@click.option(
    "--model",
    "-m",
    type=click.Choice(["st", "hf"]),
    help="st - sentence transformer, hf - huggingface model for sequence classication",
    required=True,
)
@click.option("--bert", "-b", help="BERT base", required=True)
@click.option("--epochs", "-e", help="Number of epochs", default=20)
def main(train, val, pred, checkpoint, model, bert, epochs):
    torch.seed()

    if model == "st":
        train_sentence_transformers(train, val, pred, checkpoint, bert, epochs)
    elif model == "hf":
        train_huggingface(train, val, pred, checkpoint, bert, epochs)


if __name__ == "__main__":
    main()

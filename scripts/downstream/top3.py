import click
import torch

from scripts.downstream.top3_model.cst_hf import train_huggingface


@click.command()
@click.option("--train", "-t", help="Train set jsonl path", required=True)
@click.option("--val", "-v", help="Validation set jsonl file", required=True)
@click.option("--pred", "-p", help="Predictions csv")
@click.option("--checkpoint", "-c", help="Path to store model", required=True)
@click.option("--bert", "-b", help="BERT base", required=True)
@click.option("--epochs", "-e", help="Number of epochs", default=20)
def main(train, val, pred, checkpoint, bert, epochs):
    torch.seed()

    train_huggingface(train, val, pred, checkpoint, bert, epochs)


if __name__ == "__main__":
    main()

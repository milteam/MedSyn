import click
from colorama import Fore, Style
import json
import pandas as pd
from bert_score import score


def bert_metrics(df: pd.DataFrame, left_column: str, right_column: str):
    return score(df[left_column].tolist(), df[right_column].str.replace("|", ",", regex=False).tolist(), lang="ru")


@click.command()
@click.option("--input-filepath", "-i", help="A path to an input JSONL file with model responses.", required=True)
@click.option("--taskgen-filepath", "-t", required=True)
@click.option("--output-filepath", "-o", required=True)
@click.option("--symptoms-recall-threshold", "-srt", default=0.65, required=False)
@click.option("--anamnesis-precision-threshold", "-apt", default=0.7, required=False)
def main(taskgen_filepath: str, input_filepath: str, output_filepath: str,
    symptoms_recall_threshold: float,
    anamnesis_precision_threshold: float):
    print(f"\033[34mSymptoms recall threshold = {symptoms_recall_threshold:.2f}\033[0m")
    print(f"\033[34mAnamnesis precision threshold = {anamnesis_precision_threshold:.2f}\033[0m")
    df = pd.read_csv(taskgen_filepath)

    with open(input_filepath, "rt", encoding="utf-8") as file:
        df["response"] = [json.loads(line)["answer"] for line in file]

    _, r, _ = bert_metrics(df, "response", "shuffled_symptoms")
    p, _, _ = bert_metrics(df, "response", "anamnesis")

    df["symptoms_recall"] = r.cpu().numpy()
    df["anamnesis_precision"] = p.cpu().numpy()
    df["FILTERING_OK"] = (df["symptoms_recall"] >= symptoms_recall_threshold) & (df["anamnesis_precision"] > anamnesis_precision_threshold)

    df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    main()
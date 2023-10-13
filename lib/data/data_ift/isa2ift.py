"""Convert isa anamnesis data to IFT data format."""

from typing import Dict

import os
import re
import random
from tqdm import tqdm
import pandas as pd
import json
import click


INSTRUCTIONS = [
    "Допиши анамнез",
    "Продолжи анамнез",
    "Закончи анамнез",
    "Напиши продолжение анамнеза",
]


def get_sample_by_continuation(text: str, ratio: int = 2) -> Dict:
    instruction = random.choice(INSTRUCTIONS)

    output = text.replace("\n", " ")
    output = re.sub(" +", " ", output).split(" ")
    pre = " ".join(output[: len(output) // ratio])
    post = " ".join(output[len(output) // ratio :])

    sample = {"instruction": instruction, "input": pre, "output": post}
    return sample


@click.command()
@click.option("--results-dir", default="data/data_ift/isa")
@click.option("--result-name", default="isa_data_ift.json")
@click.option("--samples-path", default="data/data_raw/isa_anamnesis.csv")
def generate_data(results_dir: str, result_name: str, samples_path: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []
    samples = pd.read_csv(samples_path)
    samples.dropna(axis=0, inplace=True)

    for idx in tqdm(samples.index):
        text = samples.loc[idx]["text"]

        new_sample = get_sample_by_continuation(text, ratio=2)
        result.append(new_sample)

        new_sample = get_sample_by_continuation(text, ratio=3)
        result.append(new_sample)

        new_sample = get_sample_by_continuation(text, ratio=4)
        result.append(new_sample)

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

"""Convert diagnoz data to IFT data format."""

from typing import Dict

import os
import re
from tqdm import tqdm
import pandas as pd
import json
import click


def get_sample(symptoms: str, disease: str) -> Dict:
    instruction = "Напиши название заболевания по симптомам."
    symptoms = ", ".join(symptoms.split(","))
    symptoms = ", ".join(symptoms.split("."))
    input = f"Симптомы: {symptoms}"
    sample = {"instruction": instruction, "input": input, "output": disease}
    return sample


def get_sample_by_continuation(symptoms: str, disease: str) -> Dict:
    instruction = f"Допиши симптомы для заболевания {disease}"

    symptoms = symptoms.replace(",", ".").split(".")
    pre_out = ", ".join(symptoms[: len(symptoms) // 2])
    output_sample = ", ".join(symptoms[len(symptoms) // 2 :])

    pre_out = f"Симптомы: {pre_out}"

    sample = {"instruction": instruction, "input": pre_out, "output": output_sample}
    return sample


@click.command()
@click.option("--results-dir", default="data/data_ift/diagnoz")
@click.option("--result-name", default="diagnoz_data_ift.json")
@click.option("--samples-path", default="data/data_raw/diagnoz.csv")
def generate_data(results_dir: str, result_name: str, samples_path: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []
    samples = pd.read_csv(samples_path, sep=";")

    for idx in tqdm(samples.index):
        disease = samples.loc[idx]["диагноз"]
        symptoms = samples.loc[idx]["симптомы"]

        new_sample = get_sample(symptoms, disease)
        result.append(new_sample)

        new_sample = get_sample_by_continuation(symptoms, disease)
        result.append(new_sample)

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

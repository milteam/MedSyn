"""Convert generated anamnesis from second iteration to LLaMA IFT data format."""

from typing import Dict

import os
import re
import random
import pandas as pd
from os.path import join
from tqdm import tqdm
import json
import click

PRE = [
    "",
    "Ты являешься врачом.",
    "Ты являешься профессиональным врачом.",
]

INSTRUCTIONS = [
    "Составь описание анамнеза, основанного на клинических проявлениях пациента, зафиксированных врачом.",
    "Формулируй запись истории болезни, которую врач подготовил на основе симптомов пациента.",
    "Создай детализированный анамнез, отражающий симптоматику пациента, как это определил медицинский специалист.",
    "Запиши медицинский анамнез, включающий симптомы пациента, описанные врачом.",
    "Произведи текст анамнеза, который был бы сформирован врачом на основе наблюдаемых симптомов пациента."
]

def get_sample(data: pd.Series) -> Dict:
    instruction = data["instruction"]
    input = data["prompt"]
    output = data["response"]
    sample = {"instruction": instruction, "input": input, "output": output}
    return sample


def get_sample_with_sympt(data: pd.Series, task: pd.DataFrame) -> Dict:
    id = data["sample_id"]
    initial_data = task.query("sample_id == @id")
    symptoms = initial_data["shuffled_symptoms"].values[0]
    symptoms = symptoms.replace("|", ",")

    instruction = random.choice(INSTRUCTIONS)
    pre = random.choice(PRE)
    instruction = pre + " " + instruction if pre else instruction

    output = data["response"]
    sample = {"instruction": instruction, "input": symptoms, "output": output}
    return sample


@click.command()
@click.option("--input-dir", default="data/data_raw/gpt_v1")
@click.option("--result-name", default="gpt_generated_v1.json")
@click.option("--results-dir", default="data/data_ift/gpt_generated_v1")
def generate_data(input_dir: str, results_dir: str, result_name: str):
    os.makedirs(results_dir, exist_ok=True)

    result = []

    task = pd.read_csv(join(input_dir, "sampled_taskgen.csv"))
    gpt_data = pd.read_csv(join(input_dir, "gpt4-generated.csv"))

    for idx in range(len(gpt_data)):
        record = gpt_data.iloc[idx]

        sample = get_sample(record)
        result.append(sample)

        sample = get_sample_with_sympt(record, task)
        result.append(sample)

    print(f"{len(result)} samples generated.")

    with open(join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

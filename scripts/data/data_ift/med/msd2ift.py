"""Convert MSD data to IFT data format."""

from typing import Dict

import os
import json
import click
import random
from tqdm import tqdm

PRE = [
    "",
    "Ты эксперт в обласи медицины.",
    "Ты являешься профессиональным врачом.",
]


INSTRUCTIONS = [
    "Напиши медицинскую информацю по теме {}.",
    "Предоставьте медицинские сведения на тему {}.",
    "Опишите медицинские данные, связанные с темой {}.",
    "Сформулируйте медицинскую информацию, касающуюся {}.",
    "Изложите медицинские факты по вопросу {}.",
    "Произведите медицинский обзор на тематику {}.",
]


@click.command()
@click.option("--results-dir", default="data/data_ift/msd")
@click.option("--result-name", default="msd_symptoms.json")
@click.option("--data-path", default="data/data_raw/msd/msd_symptoms.json")
def generate_data(results_dir: str, result_name: str, data_path: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []
    with open(data_path, "rb") as f:
        data = json.load(f)

    for topic, data in tqdm(data.items()):
        for sub_topic, sub_topic_data in data.items():
            instruction = random.choice(INSTRUCTIONS).format(topic)
            pre = random.choice(PRE)
            instruction = pre + " " + instruction if pre else instruction

            sample = {
                "instruction": instruction,
                "input": sub_topic,
                "output": sub_topic_data,
            }

            result.append(sample)

    print(f"{len(result)} samples generated.")

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

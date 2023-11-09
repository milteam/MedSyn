"""Convert Wikipedia data to IFT data format."""

from typing import Dict

import os
import re
import random
from tqdm import tqdm
import pandas as pd
import json
import click


INSTRUCTIONS_1 = [
    "Напиши название заболевания к которому относится текст",
    "Укажи болезнь, описываемую в тексте.",
    "Определи заболевание, на которое указывает данный текст.",
    "Назови болезнь, связанную с этим текстом.",
    "Опиши диагноз, соответствующий содержанию текста.",
    "Опиши название недуга, о котором идёт речь в тексте.",
]

INSTRUCTIONS_2 = [
    "Дополни текст, касающийся этого недуга."
    "Продолжи описание, связанное с болезнью."
    "Расширь изложение, относящееся к данному заболеванию."
    "Заверши написание текста о данной болезни."
    "Продлжи развернутое изложение, касающееся этого медицинского состояния."
]


def get_sample(text: str, disease: str) -> Dict:
    instruction = random.choice(INSTRUCTIONS_1)
    sample = {"instruction": instruction, "input": text, "output": disease}
    return sample


def get_sample_by_continuation(text: str, disease: str) -> Dict:
    instruction_ = random.choice(INSTRUCTIONS_2)
    instruction = f"{instruction_} {disease}"

    output = text.replace("\n", " ")
    output = re.sub(" +", " ", output).split(" ")
    pre_out = " ".join(output[len(output) // 2 :])
    output_sample = " ".join(output[: len(output) // 2])

    sample = {"instruction": instruction, "input": output_sample, "output": pre_out}
    return sample


@click.command()
@click.option("--results-dir", default="data/data_ift/wikipedia")
@click.option("--result-name", default="wikipedia_data_ift.json")
@click.option("--samples-path", default="data/data_raw/wiki_disease_ru.csv")
def generate_data(results_dir: str, result_name: str, samples_path: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []
    samples = pd.read_csv(samples_path)

    for idx in tqdm(samples.index):
        disease = samples.loc[idx]["Index"]
        text = samples.loc[idx]["Симптомы"]

        new_sample = get_sample(text, disease)
        result.append(new_sample)

        new_sample = get_sample_by_continuation(text, disease)
        result.append(new_sample)

    print(f"{len(result)} samples generated.")

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

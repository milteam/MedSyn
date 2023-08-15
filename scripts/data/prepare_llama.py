"""Convert generated anamnesis to LLaMA IFT data format."""

from typing import Dict

import os
from os.path import join
import re
from tqdm import tqdm
import json
import click

INSTRUCTION = "Напиши текст анамнеза, составленного врачом по итогам приема пациента."


def get_sample(data: Dict) -> Dict:
    symptoms = data["symptoms"]
    gender = data["gender"]
    marital_state = data["marital_state"]
    smoking = data["smoking"]

    if gender == "male":
        marital = "замужний" if marital_state else "незамужний"
        smoking = "курящий" if smoking else "некурящий"
        gender_ru = "мужчина"
        conj = "который"
    else:
        marital = "замужняя" if marital_state else "незамужняя"
        smoking = "курящая" if smoking else "некурящая"
        gender_ru = "женщина"
        conj = "которая"

    symptoms = ", ".join(symptoms).lower()
    input = f"Пациент - {marital} {smoking} {gender_ru}, {conj} жалуется на {symptoms}."

    output = data["response"].replace("\n", " ")
    output = re.sub(" +", " ", output)

    sample = {"instruction": INSTRUCTION, "input": input, "output": output}
    return sample


@click.command()
@click.option(
    "--results-dir", default="/home/gleb/VSProjects/projects/MIL/SberMedText/results"
)
@click.option("--result-name", default="alpaca_med_data.json")
def generate_data(results_dir: str, result_name: str):
    result = []

    generated_samples = os.listdir(results_dir)

    for single_file in tqdm(generated_samples):
        if single_file != result_name:
            with open(join(results_dir, single_file), "rb") as f:
                samples = json.load(f)
            for data in samples:
                new_sample = get_sample(data)
                result.append(new_sample)

    with open(join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

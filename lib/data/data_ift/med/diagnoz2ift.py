"""Convert diagnoz data to IFT data format."""

from typing import Dict

import os
import json
import click
import random
from tqdm import tqdm
import pandas as pd

PRE = [
    "",
    "Ты являешься профессиональным врачом.",
    "Ты являешься врачом.",
    "Твоя профессия - врач.",
    "Ты обладаешь квалификацией профессионального медика.",
    "Ты - специалист в области медицины.",
    "Ты являешься квалифицированным медицинским работником.",
    "Ты занимаешься профессиональной медицинской деятельностью.",
]

INSTRUCTIONS_1 = [
    "Напиши название заболевания по симптомам.",
    "Определи болезнь на основе её симптомов.",
    "Укажи диагноз, исходя из описанных признаков.",
    "Предложи название болезни, соответствующее симптомам.",
    "Сформулируй диагноз, опираясь на перечисленные симптомы.",
    "Выдели заболевание, основываясь на его симптоматике.",
]

INSTRUCTIONS_2 = [
    "Напиши симптомы для заболевания.",
    "Перечисли признаки, характерные для болезни.",
    "Опиши симптоматику, связанную с этим заболеванием.",
    "Укажи характерные симптомы данного заболевания.",
    "Выдели основные проявления этой болезни.",
    "Задокументируй типичные признаки для данного заболевания.",
]


def get_sample(symptoms: str, disease: str) -> Dict:
    instruction = random.choice(INSTRUCTIONS_1)
    pre = random.choice(PRE)
    instruction = pre + " " + instruction if pre else instruction

    symptoms = ", ".join(symptoms.split(","))
    symptoms = ", ".join(symptoms.split("."))
    input = f"Симптомы: {symptoms}"
    sample = {"instruction": instruction, "input": input, "output": disease}
    return sample


def get_symptoms_sample(symptoms: str, disease: str) -> Dict:
    instruction = random.choice(INSTRUCTIONS_2)
    pre = random.choice(PRE)
    instruction = pre + " " + instruction if pre else instruction

    sample = {"instruction": instruction, "input": disease, "output": symptoms}
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

        new_sample = get_symptoms_sample(symptoms, disease)
        result.append(new_sample)

    print(f"{len(result)} samples generated.")

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

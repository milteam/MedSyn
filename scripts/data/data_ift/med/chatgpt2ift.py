"""Convert generated anamnesis from first iteration to LLaMA IFT data format."""

from typing import Dict

import os
import random
from os.path import join
import re
from tqdm import tqdm
import json
import click

PRE = [
    "",
    "Ты являешься профессиональным врачом.",
    "Ты являешься врачом.",
    "Ты обладаешь квалификацией профессионального медика.",
    "Ты - специалист в области медицины.",
]

INSTRUCTIONS = [
    "Напиши текст анамнеза, составленного врачом по итогам приема пациента.",
    "Сформулируй запись анамнеза на основе данных, полученных врачом во время консультации пациента.",
    "Создай текстовое описание истории болезни, подготовленное доктором после осмотра пациента.",
    "Оформи документацию медицинского анамнеза, разработанную врачом после визита пациента.",
    "Составь изложение анамнеза, которое врач заполняет по результатам приема пациента.",
    "Запиши историю заболевания пациента, как это делает врач по завершении медицинского осмотра.",
]


def get_sample(data: Dict) -> Dict:
    instruction = random.choice(INSTRUCTIONS)
    pre = random.choice(PRE)

    instruction = pre + " " + instruction if pre else instruction

    symptoms = data["symptoms"]
    gender = data["gender"]
    marital_state = data["marital_state"]
    smoking = data["smoking"]

    if gender == "male":
        marital = "женатый" if marital_state else "неженатый"
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

    sample = {"instruction": instruction, "input": input, "output": output}
    return sample


@click.command()
@click.option("--input-dir", default="data/data_raw/chatgpt_generated")
@click.option("--result-name", default="gpt_generated.json")
@click.option("--results-dir", default="data/data_ift/chatgpt_generated")
def generate_data(input_dir: str, results_dir: str, result_name: str):
    os.makedirs(results_dir, exist_ok=True)

    result = []

    generated_dirs = os.listdir(input_dir)

    for single_dir in tqdm(generated_dirs):
        for file in os.listdir(join(input_dir, single_dir)):
            with open(join(input_dir, single_dir, file), "rb") as f:
                samples = json.load(f)
            for data in samples:
                new_sample = get_sample(data)
                result.append(new_sample)

    print(f"{len(result)} samples generated.")

    with open(join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

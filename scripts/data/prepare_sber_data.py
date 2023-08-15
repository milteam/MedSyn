"""Convert real anamnesis to LLaMA IFT data format."""

from typing import Dict, Set

from os.path import join
import re
from tqdm import tqdm
import pandas as pd
import json
import click

INSTRUCTION = "Напиши текст анамнеза, составленного врачом по итогам приема пациента."


def get_sample(age: int, text: str, gender: int, symptoms_set: Set) -> Dict:
    gender = "male" if gender == 1 else "female"
    symptoms = get_symptoms_form_text(text, symptoms_set)

    if gender == "male":
        gender_ru = "мужчина"
        conj = "который"
    else:
        gender_ru = "женщина"
        conj = "которая"

    if symptoms:
        symptoms = ", ".join(symptoms).lower()
        input = f"Пациент - {gender_ru} в возрасте {age} лет, {conj} жалуется на {symptoms}."
    else:
        input = f"Пациент - {gender_ru} в возрасте {age} лет."

    output = text.replace("\n", " ")
    output = re.sub(" +", " ", output)

    sample = {"instruction": INSTRUCTION, "input": input, "output": output}
    return sample


def get_symptoms_form_text(text: str, symptoms_set: Set):
    symptoms_in_text = set()
    for symp in symptoms_set:
        if symp in text.lower():
            symptoms_in_text.add(symp)
    return symptoms_in_text


def get_symptoms(symptoms_path: str) -> Set:
    symptoms_mapping = pd.read_csv(symptoms_path)
    symptoms_set = set()
    for s in symptoms_mapping["symp_ru"].values:
        for symp in s.split(","):
            symptoms_set.add(symp.lower().strip())
    return symptoms_set


@click.command()
@click.option("--results-dir", default="results")
@click.option("--result-name", default="alpaca_sber_data.json")
@click.option("--samples-path", default="data/samples.csv")
@click.option("--symptoms-path", default="data/eng_to_ru/symptoms_eng_ru.csv")
def generate_data(
    results_dir: str, result_name: str, symptoms_path: str, samples_path: str
):
    result = []
    samples = pd.read_csv(samples_path)
    symptoms_set = get_symptoms(symptoms_path)

    for idx in tqdm(samples.index):
        age = samples.loc[idx]["age"]
        gender = samples.loc[idx]["gender"]
        text = samples.loc[idx]["raw_visit_text"]

        new_sample = get_sample(age, text, gender, symptoms_set)
        result.append(new_sample)

    with open(join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

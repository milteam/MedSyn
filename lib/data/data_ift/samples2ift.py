"""Convert real anamnesis to IFT data format."""

from typing import Dict, Set, List

import os
import re
import random
from tqdm import tqdm
import pandas as pd
import json
import click

INSTRUCTIONS_1 = [
    "Напиши текст анамнеза, составленного врачом по итогам приема пациента.",
    "Напиши текст анамнеза, составленного врачом по со слов пациента.",
    "Ты являешься врачом. Напиши анамнез, составленный со слов пациента.",
    "Сформулируй анамнез на основе информации, полученной во время консультации с пациентом.",
    "Запиши детальный анамнез, используя данные, предоставленные пациентом во время осмотра.",
    "Как медицинский специалист, оформи анамнез, исходя из беседы с пациентом.",
    "Опиши историю болезни пациента, основываясь на его личных рассказах и жалобах.",
    "В роли врача, составь подробный анамнез, собрав всю необходимую информацию от пациента.",
    "Как клиницист, документируй анамнез, включая все детали, упомянутые пациентом.",
]

INSTRUCTIONS_2 = [
    "Допиши анамнез",
    "Продолжи анамнез",
    "Закончи анамнез",
    "Напиши продолжение анамнеза",
    "Напиши продолжение истории болезни",
    "Напиши продолжение",
    "Заверши начатый анамнез",
    "Продлжи описание истории болезни",
    "Добавь детали к анамнезу",
    "Добавь продолжение",
]


def get_sample_by_symptoms(age: int, text: str, gender: int, symptoms: List) -> Dict:
    gender = "male" if gender == 1 else "female"

    if gender == "male":
        gender_ru = "мужчина"
        conj = "который"
    else:
        gender_ru = "женщина"
        conj = "которая"

    symptoms = ", ".join(symptoms).lower()
    input = (
        f"Пациент - {gender_ru} в возрасте {age} лет, {conj} жалуется на {symptoms}."
    )

    output = text.replace("\n", " ")
    output = re.sub(" +", " ", output)
    instruction = random.choice(INSTRUCTIONS_1)

    sample = {"instruction": instruction, "input": input, "output": output}
    return sample


def get_sample_by_code(age: int, text: str, gender: int, disease_name: str) -> Dict:
    gender = "male" if gender == 1 else "female"

    if gender == "male":
        gender_ru = "мужчина"
        conj = "который"
    else:
        gender_ru = "женщина"
        conj = "которая"

    input = f"Пациент - {gender_ru} в возрасте {age} лет c заболеванием {disease_name}."

    output = text.replace("\n", " ")
    output = re.sub(" +", " ", output)
    instruction = random.choice(INSTRUCTIONS_1)

    sample = {"instruction": instruction, "input": input, "output": output}
    return sample


def get_sample_by_continuation(
    age: int,
    text: str,
    gender: int,
) -> Dict:
    instruction = random.choice(INSTRUCTIONS_2)

    gender = "male" if gender == 1 else "female"

    if gender == "male":
        gender_ru = "мужчина"
    else:
        gender_ru = "женщина"

    output = text.replace("\n", " ")
    output = re.sub(" +", " ", output).split(" ")
    pre_out = " ".join(output[len(output) // 2 :])
    output_sample = " ".join(output[: len(output) // 2])

    input = (
        f"Пациент - {gender_ru} в возрасте {age} лет. Начало анамнеза: {output_sample}"
    )

    sample = {"instruction": instruction, "input": input, "output": pre_out}
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
@click.option("--min-words", default=12)
@click.option("--results-dir", default="data/data_ift/samples")
@click.option("--result-name", default="samples_data_ift.json")
@click.option("--samples-path", default="data/data_raw/samples.csv")
@click.option("--codes-path", default="data/data_raw/spr_mkb10.csv")
@click.option("--symptoms-path", default="data/data_raw/symptoms_eng_ru.csv")
def generate_data(
    min_words: int,
    results_dir: str,
    result_name: str,
    samples_path: str,
    codes_path: str,
    symptoms_path: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    symptoms_set = get_symptoms(symptoms_path)

    result = []
    samples = pd.read_csv(samples_path)
    codes = pd.read_csv(codes_path)

    for idx in tqdm(samples.index):
        age = samples.loc[idx]["age"]
        icd_code = samples.loc[idx]["gt"]
        gender = samples.loc[idx]["gender"]
        text = samples.loc[idx]["raw_visit_text"]

        if len(text.split(" ")) > min_words:

            symptoms = get_symptoms_form_text(text, symptoms_set)
            if symptoms:
                new_sample_symp = get_sample_by_symptoms(age, text, gender, symptoms)
                result.append(new_sample_symp)

            dis_name = codes.query("mkb_cod == @icd_code")
            if dis_name.shape[0] > 0:
                disease_name = dis_name["mkb_name"].values[0]
                new_sample_code = get_sample_by_code(age, text, gender, disease_name)
                result.append(new_sample_code)

            new_sample_continue = get_sample_by_continuation(age, text, gender)
            result.append(new_sample_continue)

    print(f"{len(result)} samples generated.")

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

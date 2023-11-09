"""Convert MedPrime data to IFT data format."""

from typing import Dict, Set

import os
import re
import random
from tqdm import tqdm
import pandas as pd
import json
import click

INSTRUCTIONS = [
    "Напиши текст анамнеза, составленного врачом по итогам приема пациента.",
    "Напиши текст анамнеза, составленного врачом по со слов пациента.",
    "Ты являешься врачом. Напиши анамнез, составленный со слов пациента.",
    "Оформи анамнез, как если бы ты был врачом, записывающим информацию после осмотра пациента.",
    "Запиши историю болезни, основываясь на описании симптомов пациентом.",
    "В качестве медицинского специалиста, составь анамнез, опираясь на данные, предоставленные пациентом.",
    "Создай текст анамнеза, будучи врачом, который анализирует информацию, полученную от пациента.",
    "Как врач, документируй анамнез, используя информацию, изложенную пациентом.",
    "От имени медицинского работника, составь подробный анамнез на основе рассказа пациента.",
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

INSTRUCTIONS_SYMP = [
    "Допиши симптомы",
    "Допиши релевантные симптомы",
    "Напиши какие еще симптомы могут встретиться вместе с этими",
    "Продолжи список симптомов",
    "Добавь дополнительные признаки заболевания",
    "Перечисли другие потенциальные симптомы",
    "Расширь список характерных признаков",
    "Заверши описание всех возможных симптомов",
    "Укажи дополнительные проявления болезни",
    "Дополни перечень симптоматики этого состояния",
    "Опиши другие признаки, которые могут сопровождать указанные симптомы",
]

PRE = ["Пациент жалуется", "Пациент обратился с", "Жалоба пациента", ""]


def get_sample_by_continuation(text: str, ratio: int = 2, mode: str = "anam") -> Dict:
    if mode == "symptoms":
        instruction = random.choice(INSTRUCTIONS_SYMP)
    else:
        instruction = random.choice(INSTRUCTIONS_2)

    output = text.replace("\n", " ")

    output = re.sub(" +", " ", output).split(" ")
    pre = " ".join(output[: len(output) // ratio])
    post = " ".join(output[len(output) // ratio :])

    if mode == "symptoms":
        if pre.split(" ")[0] not in ["Жалобы", "жалобы", "жалуется"]:
            pre_ = random.choice(PRE)
            pre = f"{pre_} {pre}"

    sample = {"instruction": instruction, "input": pre, "output": post}
    return sample


def get_sample(symptoms: str, anamnesis: str) -> Dict:
    if symptoms.split(" ")[0] == "Жалобы":
        input = f"{symptoms}"
    else:
        pre = random.choice(PRE)
        input = f"{pre} {symptoms}"

    output = anamnesis.replace("\n", " ")
    output = re.sub(" +", " ", output)

    instruction = random.choice(INSTRUCTIONS)
    sample = {"instruction": instruction, "input": input, "output": output}
    return sample


@click.command()
@click.option("--min-words", default=20)
@click.option("--results-dir", default="data/data_ift/rumedprime")
@click.option("--result-name", default="medprime_data_ift.json")
@click.option("--samples-path", default="data/data_raw/RuMedPrimeData.tsv")
def generate_data(
    results_dir: str, result_name: str, samples_path: str, min_words: int
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []
    samples = pd.read_csv(samples_path, sep="\t")

    for idx in tqdm(samples.index):
        symptoms = samples.loc[idx]["symptoms"]
        anamnesis = samples.loc[idx]["anamnesis"]

        if len(symptoms.split(" ")) + len(anamnesis.split(" ")) > min_words:
            new_sample = get_sample(symptoms, anamnesis)
            result.append(new_sample)

            new_sample = get_sample_by_continuation(symptoms, ratio=2, mode="symptoms")
            result.append(new_sample)

            text = symptoms + " " + anamnesis

            new_sample = get_sample_by_continuation(text, ratio=2)
            result.append(new_sample)

            new_sample = get_sample_by_continuation(text, ratio=3)
            result.append(new_sample)

            new_sample = get_sample_by_continuation(text, ratio=4)
            result.append(new_sample)

    print(f"{len(result)} samples generated.")

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

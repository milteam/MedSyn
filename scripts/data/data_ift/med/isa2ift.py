"""Convert isa anamnesis data to IFT data format."""

from typing import Dict

import os
import re
import random
from tqdm import tqdm
import pandas as pd
import json
import click


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


INSTRUCTIONS = [
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


def get_sample_by_continuation(text: str, ratio: int = 2) -> Dict:
    instruction = random.choice(INSTRUCTIONS)
    pre = random.choice(PRE)
    instruction = pre + " " + instruction if pre else instruction

    output = text.replace("\n", " ")
    output = re.sub(" +", " ", output).split(" ")
    pre = " ".join(output[: len(output) // ratio])
    post = " ".join(output[len(output) // ratio :])

    sample = {"instruction": instruction, "input": pre, "output": post}
    return sample


@click.command()
@click.option("--min-words", default=12)
@click.option("--results-dir", default="data/data_ift/isa")
@click.option("--result-name", default="isa_data_ift.json")
@click.option("--samples-path", default="data/data_raw/isa_anamnesis.csv")
def generate_data(
    results_dir: str, result_name: str, samples_path: str, min_words: int
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []
    samples = pd.read_csv(samples_path)
    samples.dropna(axis=0, inplace=True)

    for idx in tqdm(samples.index):
        text = samples.loc[idx]["text"]

        if len(text.split(" ")) > min_words:

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

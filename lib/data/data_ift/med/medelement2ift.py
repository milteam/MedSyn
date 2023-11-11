"""Convert medElement data to IFT data format."""

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
    "Ты являешься врачом.",
    "Твоя профессия - врач.",
    "Ты обладаешь квалификацией профессионального медика.",
    "Ты - специалист в области медицины.",
    "Ты являешься квалифицированным медицинским работником.",
]


INSTRUCTIONS_1 = [
    "Напиши развернутое определение термина.",
    "Сформулируй подробное описание данного термина.",
    "Предоставь развёрнутое определение этого понятия.",
    "Изложи детальное объяснение термина.",
    "Опиши термин в полном и расширенном виде.",
    "Дай углублённое и полное определение этому слову.",
]


def get_sample(term: str, desc: str) -> Dict:
    instruction = random.choice(INSTRUCTIONS_1)
    pre = random.choice(PRE)
    instruction = pre + " " + instruction if pre else instruction

    input = term.strip()
    input = re.sub(r"(?<=\D)\d$", "", input)

    desc = desc.strip()
    desc = re.sub(r"(?<=\D)\d$", "", desc)

    sample = {"instruction": instruction, "input": input, "output": desc}
    return sample


@click.command()
@click.option("--results-dir", default="data/data_ift/medElement")
@click.option("--result-name", default="medElement_data_ift.json")
@click.option("--samples-path", default="data/data_raw/med_element.csv")
def generate_data(results_dir: str, result_name: str, samples_path: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []
    samples = pd.read_csv(samples_path)

    for idx in tqdm(samples.index):
        term = samples.loc[idx]["Term"]
        desc = samples.loc[idx]["Description"]

        new_sample = get_sample(term, desc)
        result.append(new_sample)

    print(f"{len(result)} samples generated.")

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

"""Convert WikiMed data to LLaMA IFT format."""

from typing import Dict

import os
import re
from os.path import join
from tqdm import tqdm
import json
import click
import pandas as pd

INSTRUCTIONS_RU = {
    "Этиология и патогенез": "Напиши описание патогенеза заболевания",
    "Клинические проявления": "Напиши клинические проявления заболевания",
    "Диагностика": "Напиши диагностику заболевания",
    "Дифференциальный диагноз": "Напиши дифференциальный диагноз заболевания",
    "Лечение": "Напиши лечение заболевания",
    "Профилактика": "Напиши профилактику заболевания",
}
INSTRUCTIONS_ENG = {
    "Этиология и патогенез": "pathogenesis",
    "Клинические проявления": "manifestations",
    "Диагностика": "diagnostics",
    "Дифференциальный диагноз": "differential",
    "Лечение": "treatment",
    "Профилактика": "prevention",
}


def prepare_name(disease_name: str) -> str:
    disease_name = disease_name.replace(
        "Злокач. новообр.", "Злокачественное новообразование"
    )
    disease_name = disease_name.replace("вызв.", "вызванный")

    disease_name = (
        disease_name.removeprefix("Другие")
        .replace(", не классифицированные в других рубриках", "")
        .strip()
    )
    # Remove ICD code from name:
    disease_name = re.sub(r"\([A-Z][0-9]+(\.[0-9\-]+)?[\+\*]?\)", "", disease_name)
    return disease_name


def get_sample(data: pd.Series, instruction_type: str, codes: pd.DataFrame) -> Dict:
    icd_code = data["МКБ-10"]
    dis_name = codes.query("mkb_cod == @icd_code")
    if dis_name.shape[0] > 0:
        name = dis_name["mkb_name"].values[0]
    else:
        name = data["Рубрика"]

    input = prepare_name(name)
    instruction = INSTRUCTIONS_RU[instruction_type]
    output = data[instruction_type]
    output = output.replace("\n", " ")
    sample = {"instruction": instruction, "input": input, "output": output}
    return sample


@click.command()
@click.option("--wikimed-path", default="data/wikimed/wikimed_diseases.csv")
@click.option("--codes-path", default="data/eng_to_ru/spr_mkb10.csv")
@click.option(
    "--results-dir",
    default="/home/gleb/VSProjects/projects/MIL/SberMedText/results/wikimed",
)
def generate_data(wikimed_path: str, codes_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    codes = pd.read_csv(codes_path)
    wikimed = pd.read_csv(wikimed_path)

    for instruction_type in tqdm(INSTRUCTIONS_RU.keys()):
        result = []

        wikimed_data = wikimed.dropna(subset=[instruction_type], axis=0, inplace=False)
        for idx in wikimed_data.index:
            data = wikimed_data.loc[idx]
            sample = get_sample(data, instruction_type, codes)
            result.append(sample)

        result_name = INSTRUCTIONS_ENG[instruction_type]
        with open(join(results_dir, result_name + ".json"), "w", encoding="utf8") as f:
            json.dump(result, f, indent=3, ensure_ascii=False)

        print(f"Prepared {len(result)} samples for {result_name}.")


if __name__ == "__main__":
    generate_data()

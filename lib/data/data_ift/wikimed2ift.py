"""Convert WikiMed data to IFT format."""

from typing import Dict

import os
import re
from os.path import join
from tqdm import tqdm
import json
import click
import pandas as pd


INSTRUCTIONS_DIS_RU = {
    "Этиология и патогенез": "Напиши описание патогенеза заболевания",
    "Клинические проявления": "Напиши клинические проявления заболевания",
    "Диагностика": "Напиши диагностику заболевания",
    "Дифференциальный диагноз": "Напиши дифференциальный диагноз заболевания",
    "Лечение": "Напиши лечение заболевания",
    "Профилактика": "Напиши профилактику заболевания",
    "Действующие вещества": "Напиши действующие вещества для заболевания",
}
INSTRUCTIONS_DIS_ENG = {
    "Этиология и патогенез": "pathogenesis",
    "Клинические проявления": "manifestations",
    "Диагностика": "diagnostics",
    "Дифференциальный диагноз": "differential",
    "Лечение": "treatment",
    "Профилактика": "prevention",
    "Действующие вещества": "acting_meds",
}


INSTRUCTIONS_DRG_RU = {
    "Фармакологическая группа": "Напиши фармакологическую группу вещества",
    "Характеристика вещества": "Напиши характеристики вещества",
    "Фармакология": "Опиши фармакологию вещества",
    "Применение": "Напиши способы применения вещества",
    "Противопоказания": "Перечисли противопоказания вещества",
    "Побочные действия": "Перечисли побочные действрия вещества",
    "Взаимодействие": "Опиши взаимодействие вещества с другими",
    "Способ применения и дозы": "Напиши способ применения и дозы для вещества",
    "Меры предосторожности": "Опиши меры предосторожности для вещества",
    "МКБ-10": "Перечисли заболевания для которых может применяться вещество",
}
INSTRUCTIONS_DRG_ENG = {
    "Фармакологическая группа": "drug_pharm_group",
    "Характеристика вещества": "drug_char",
    "Фармакология": "drug_pharmacology",
    "Применение": "drug_usage",
    "Противопоказания": "drug_neg",
    "Побочные действия": "drug_neg_symp",
    "Взаимодействие": "drug_connection",
    "Способ применения и дозы": "drug_dosage",
    "Меры предосторожности": "dgug_cautiuons",
    "МКБ-10": "drug_diseases",
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
    instruction = INSTRUCTIONS_DIS_RU[instruction_type]
    output = data[instruction_type]
    output = output.replace("\n", " ")
    sample = {"instruction": instruction, "input": input, "output": output}
    return sample


def get_drug_sample(data: pd.Series, instruction_type: str) -> Dict:
    input = data["Название"]
    instruction = INSTRUCTIONS_DRG_RU[instruction_type]
    output = data[instruction_type]
    output = output.replace("\n", " ")

    sample = {"instruction": instruction, "input": input, "output": output}
    return sample


def generate_data_for_diseases(wikimed_path: str, codes_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    codes = pd.read_csv(codes_path)
    wikimed = pd.read_csv(wikimed_path)

    for instruction_type in tqdm(INSTRUCTIONS_DIS_RU.keys()):
        result = []

        wikimed_data = wikimed.dropna(subset=[instruction_type], axis=0, inplace=False)
        for idx in wikimed_data.index:
            data = wikimed_data.loc[idx]
            sample = get_sample(data, instruction_type, codes)
            result.append(sample)

        result_name = INSTRUCTIONS_DIS_ENG[instruction_type]
        with open(join(results_dir, result_name + ".json"), "w", encoding="utf8") as f:
            json.dump(result, f, indent=3, ensure_ascii=False)

        print(f"Prepared {len(result)} samples for {result_name}.")


def generate_data_for_drugs(wikimed_drugs_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)

    wikimed = pd.read_csv(wikimed_drugs_path)

    for instruction_type in tqdm(INSTRUCTIONS_DRG_RU.keys()):
        result = []

        wikimed_data = wikimed.dropna(subset=[instruction_type], axis=0, inplace=False)
        for idx in wikimed_data.index:
            data = wikimed_data.loc[idx]
            sample = get_drug_sample(data, instruction_type)
            result.append(sample)

        result_name = INSTRUCTIONS_DRG_ENG[instruction_type]
        with open(join(results_dir, result_name + ".json"), "w", encoding="utf8") as f:
            json.dump(result, f, indent=3, ensure_ascii=False)

        print(f"Prepared {len(result)} samples for {result_name}.")


@click.command()
@click.option("--wikimed-path", default="data/data_raw/wikimed/wikimed_diseases.csv")
@click.option("--wikimed-drugs-path", default="data/data_raw/wikimed/wikimed_meds.csv")
@click.option("--codes-path", default="data/data_raw/spr_mkb10.csv")
@click.option(
    "--results-dir",
    default="data/data_ift/wikimed/",
)
def generate_data(
    wikimed_path: str, wikimed_drugs_path: str, codes_path: str, results_dir: str
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    generate_data_for_drugs(wikimed_drugs_path, results_dir)
    generate_data_for_diseases(wikimed_path, codes_path, results_dir)


if __name__ == "__main__":
    generate_data()

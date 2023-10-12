"""Convert medElement data to IFT data format."""

from typing import Dict

import os
import re
from tqdm import tqdm
import pandas as pd
import json
import click


def get_sample(term: str, desc: str) -> Dict:
    instruction = "Напиши развернутое определение термина."

    input = f"Термин - {term}".strip()
    input = re.sub(r"(?<=\D)\d$", "", input)

    desc = desc.strip()
    desc = re.sub(r"(?<=\D)\d$", "", desc)

    sample = {"instruction": instruction, "input": input, "output": desc}
    return sample


def get_sample_by_continuation(desc: str) -> Dict:
    instruction = f"Допиши определение термина."

    desc = desc.split(" ")
    pre_out = " ".join(desc[: len(desc) // 2]).strip()
    pre_out = re.sub(",+", ",", pre_out)
    pre_out = re.sub(r"(?<=\D)\d$", "", pre_out)

    output_sample = " ".join(desc[len(desc) // 2 :]).strip()
    output_sample = re.sub(",+", ",", output_sample)
    output_sample = re.sub(r"(?<=\D)\d$", "", output_sample)

    sample = {"instruction": instruction, "input": pre_out, "output": output_sample}
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

        new_sample = get_sample_by_continuation(desc)
        result.append(new_sample)

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

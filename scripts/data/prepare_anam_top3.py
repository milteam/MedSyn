from typing import Dict

import os
from os.path import join
import re
from tqdm import tqdm
import json
import click
import pandas as pd


@click.command()
@click.option(
    "--results-dir", default="/home/gleb/VSProjects/projects/MIL/SberMedText/results"
)
@click.option("--dev-path", default="data/benchmarks/RuMedTop3/dev_v1.jsonl")
@click.option("--test-path", default="data/benchmarks/RuMedTop3/test_v1.jsonl")
@click.option("--train-path", default="data/benchmarks/RuMedTop3/train_v1.jsonl")
@click.option("--n-new-samples", default=3000)
@click.option(
    "--path-to-save",
    default="/home/gleb/VSProjects/projects/MIL/RuMedBench/data/RuMedTop3/",
)
@click.option("--result-name", default="alpaca_med_data.json")
def generate_data(
    results_dir: str,
    result_name: str,
    dev_path: str,
    test_path: str,
    train_path: str,
    n_new_samples: int,
    path_to_save: str,
):
    dev = pd.read_json(dev_path, lines=True)
    test = pd.read_json(test_path, lines=True)
    train = pd.read_json(train_path, lines=True)

    dev_codes = list(dev["code"].unique())
    test_codes = list(test["code"].unique())
    train_codes = list(train["code"].unique())

    all_codes = set(dev_codes).union(set(test_codes)).union(set(train_codes))

    generated_samples = os.listdir(results_dir)
    common_samples = []

    for single_file in tqdm(generated_samples):
        if single_file != result_name:
            with open(join(results_dir, single_file), "rb") as f:
                samples = json.load(f)
            for data in samples:
                icd = data["desease_code"].split(".")[0]
                if icd in all_codes:
                    common_samples.append(data)

    train_samples_aug = []

    n_smpls = min(n_new_samples, len(common_samples))
    for idx, sample in enumerate(common_samples[:n_smpls]):
        idx = "{:08d}".format(idx)
        symptoms = sample["response"].replace("\n", " ")
        symptoms = re.sub(" +", " ", symptoms)
        code = sample["desease_code"].split(".")[0]
        data = {"idx": idx, "symptoms": symptoms, "code": code}
        train_samples_aug.append(data)

    offset = len(train)
    for idx, sample in enumerate(train_samples_aug):
        data = list(sample.values())
        train.loc[offset + idx] = data

    data_name = f"train_v1_aug_{n_smpls/1e3:.1f}k_anam.jsonl"

    train_codes = list(train["code"].unique())
    all_codes_new = set(dev_codes).union(set(test_codes)).union(set(train_codes))
    assert all_codes == all_codes_new

    train.to_json(
        os.path.join(path_to_save, data_name),
        lines=True,
        orient="records",
        force_ascii=False,
    )


if __name__ == "__main__":
    generate_data()

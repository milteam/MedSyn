"""Add generated anamnesis to RuMedTop3 data format."""

from typing import Dict

import os
from os.path import join
import re
from tqdm import tqdm
import json
import click
import pandas as pd

FULL_TRAIN_SET = 4690


@click.command()
@click.option(
    "--results-dir", default="/home/gleb/VSProjects/projects/MIL/SberMedText/results"
)
@click.option(
    "--filtration-file",
    default="/home/gleb/VSProjects/projects/MIL/SberMedText/results/experiments/filtered.csv",
)
@click.option("--filtration", is_flag=True, show_default=True, default=True)
@click.option("--dev-path", default="data/benchmarks/RuMedTop3/dev_v1.jsonl")
@click.option("--test-path", default="data/benchmarks/RuMedTop3/test_v1.jsonl")
@click.option("--train-path", default="data/benchmarks/RuMedTop3/train_v1.jsonl")
@click.option("--ratio", default=1)
@click.option("--synthetic-only", is_flag=True, show_default=True, default=False)
@click.option(
    "--path-to-save",
    default="data/benchmarks/RuMedTop3",
)
@click.option("--result-name", default="alpaca_med_data.json")
def generate_data(
    results_dir: str,
    result_name: str,
    dev_path: str,
    test_path: str,
    train_path: str,
    ratio: float,
    path_to_save: str,
    synthetic_only: bool,
    filtration_file: str,
    filtration: bool,
):
    if filtration:
        filtration_file = pd.read_csv(filtration_file)
        valid_uids = filtration_file[filtration_file["STATUS"] == 1]["UID"].values

    dev = pd.read_json(dev_path, lines=True)
    test = pd.read_json(test_path, lines=True)
    train = pd.read_json(train_path, lines=True)

    dev_codes = list(dev["code"].unique())
    test_codes = list(test["code"].unique())
    train_codes = list(train["code"].unique())

    all_codes = set(dev_codes).union(set(test_codes)).union(set(train_codes))

    generated_samples = os.listdir(results_dir)
    generated_samples = [sample for sample in generated_samples if ".json" in sample]
    common_samples = []

    for single_file in tqdm(generated_samples):
        if single_file != result_name:
            with open(join(results_dir, single_file), "rb") as f:
                samples = json.load(f)
            for data in samples:
                if filtration:
                    if data["UID"] not in valid_uids:
                        continue
                icd = data["desease_code"].split(".")[0]
                if icd in all_codes:
                    common_samples.append(data)

    train_samples_aug = []

    # Constrained sampling:
    samples_df = pd.DataFrame(columns=["icd_full", "icd", "response"])
    for idx, sample in enumerate(common_samples):
        samples_df.loc[idx] = [
            sample["desease_code"],
            sample["desease_code"].split(".")[0],
            sample["response"],
        ]

    frac = 1 if synthetic_only else min(1, ratio * FULL_TRAIN_SET / samples_df.shape[0])
    samples_df_sampled = samples_df.groupby("icd", group_keys=False).apply(
        lambda x: x.sample(frac=frac, replace=False)
    )

    for idx in range(len(samples_df_sampled)):
        sample = samples_df_sampled.iloc[idx]
        idx = "{:08d}".format(idx)
        symptoms = sample["response"].replace("\n", " ")
        symptoms = re.sub(" +", " ", symptoms)
        code = sample["icd"]
        data = {"idx": idx, "symptoms": symptoms, "code": code}
        train_samples_aug.append(data)

    if synthetic_only:
        train = pd.DataFrame(columns=["idx", "symptoms", "code"])
        for idx, sample in enumerate(train_samples_aug):
            data = list(sample.values())
            train.loc[idx] = data
    else:
        offset = len(train)
        for idx, sample in enumerate(train_samples_aug):
            data = list(sample.values())
            train.loc[offset + idx] = data

    data_name = (
        "train_v1_synt_only_anam.jsonl"
        if synthetic_only
        else f"train_v1_aug_{ratio}_cons_anam_4_filt_{int(filtration)}.jsonl"
    )

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

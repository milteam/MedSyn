"""Convert real anamnesis samples to RuMedTop3 format."""

from typing import Dict, Set, List

import os
from os.path import join
import re
from tqdm import tqdm
import pandas as pd
import json
import click


def get_sample(data: Dict, i: int) -> Dict:
    idx = "{:08d}".format(i)
    symptoms = data["response"].replace("\n", " ")
    symptoms = re.sub(" +", " ", symptoms)
    code = data["desease_code"].split(".")[0]
    sample = [idx, symptoms, code]
    return sample


def get_generated_data(generated_dir: str) -> pd.DataFrame:
    data = pd.DataFrame(columns=["idx", "symptoms", "code"])
    i = 0
    for dir in os.listdir(generated_dir):
        for sample in os.listdir(join(generated_dir, dir)):
            with open(join(generated_dir, dir, sample), "rb") as f:
                samples = json.load(f)
                for sample in samples:
                    new_sample = get_sample(sample, i)
                    data.loc[i] = new_sample
                    i += 1
    return data


@click.command()
@click.option(
    "--results-dir",
    default="/home/gleb/VSProjects/projects/MIL/SberMedText/data/benchmarks/samples",
)
@click.option(
    "--generated-dir",
    default="/home/gleb/Datasets/SberMedGenerated/14.08.2023_backup/generated",
)
@click.option("--samples-path", default="data/samples.csv")
def generate_data(results_dir: str, samples_path: str, generated_dir: str):
    generated_data = get_generated_data(generated_dir)
    generated_data_codes = generated_data["code"].unique()
    offset = generated_data.shape[0]

    samples = pd.read_csv(samples_path)
    samples_df = pd.DataFrame(columns=["idx", "symptoms", "code"])
    i = 0
    for idx in tqdm(samples.index):
        code = samples.loc[idx]["gt"].split(".")[0]
        if code in generated_data_codes:
            text = samples.loc[idx]["raw_visit_text"]
            text = text.replace("\n", " ")
            text = re.sub(" +", " ", text)
            samples_df.loc[i] = ["{:08d}".format(i + offset), text, code]
            i += 1

    intersection_codes = samples_df["code"].unique()
    print("Total codes:", len(intersection_codes))
    generated_data = generated_data.query("code in @intersection_codes")

    dev_set = samples_df.groupby("code", group_keys=False).apply(
        lambda x: x.sample(frac=0.5, replace=False)
    )
    dev_set_idx = dev_set["idx"].values
    test_set = samples_df.query("idx not in @dev_set_idx")

    generated_data.to_json(
        join(results_dir, "train_v1.jsonl"),
        lines=True,
        orient="records",
        force_ascii=False,
    )

    dev_set.to_json(
        join(results_dir, "dev_v1.jsonl"),
        lines=True,
        orient="records",
        force_ascii=False,
    )

    test_set.to_json(
        join(results_dir, "test_v1.jsonl"),
        lines=True,
        orient="records",
        force_ascii=False,
    )


if __name__ == "__main__":
    generate_data()

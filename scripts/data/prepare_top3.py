"""Managing RuMedTop3 data."""

import os
import random
import json
import argparse
import numpy as np
import pandas as pd


def pars_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--downsampling-ratio",
        default=1,
    )
    argparser.add_argument(
        "--n-new-samples",
        default=3000,
    )
    argparser.add_argument(
        "--dev-path",
        default="data/benchmarks/RuMedTop3/dev_v1.jsonl",
    )
    argparser.add_argument(
        "--test-path",
        default="data/benchmarks/RuMedTop3/test_v1.jsonl",
    )
    argparser.add_argument(
        "--train-path",
        default="data/benchmarks/RuMedTop3/train_v1.jsonl",
    )
    argparser.add_argument(
        "--samples-path",
        default="data/generated/samples.json",
    )
    argparser.add_argument(
        "--train-aug",
        default="/home/gleb/VSProjects/projects/MIL/RuMedBench/data/RuMedTop3/",
    )
    args = argparser.parse_args()
    return args


def set_seed(seed) -> None:
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    set_seed(42)
    dev = pd.read_json(args.dev_path, lines=True)
    test = pd.read_json(args.test_path, lines=True)
    train = pd.read_json(args.train_path, lines=True)

    dev_codes = list(dev["code"].unique())
    test_codes = list(test["code"].unique())
    train_codes = list(train["code"].unique())

    all_codes = set(dev_codes).union(set(test_codes)).union(set(train_codes))

    if args.downsampling_ratio == 1:
        with open(args.samples_path, "rb") as f:
            samples = json.load(f)

        sample_codes = set()
        for sample in samples.values():
            for d in sample["disease"]:
                icd = d["icd_code"].split(".")[0]
                sample_codes.add(icd)

        common_codes = set(test_codes).intersection(sample_codes)

        common_samples = []

        for sample in samples.values():
            d = sample["disease"][0]
            if d["icd_code"].split(".")[0] in common_codes:
                common_samples.append(sample)

        train_samples = []

        n_smpls = min(args.n_new_samples, len(common_samples))
        for idx, sample in enumerate(common_samples[:n_smpls]):
            pre = np.random.choice(["", "Жалуется на "])
            idx = "{:08d}".format(idx)
            symptoms = pre + ", ".join(sample["symptoms"])
            code = sample["disease"][0]["icd_code"].split(".")[0]
            data = {"idx": idx, "symptoms": symptoms, "code": code}
            train_samples.append(data)

        pad = len(train)
        for idx, sample in enumerate(train_samples):
            data = list(sample.values())
            train.loc[pad + idx] = data

        data_name = f"train_v1_aug_{n_smpls//1e3}k.jsonl"

    else:
        train = train.groupby("code", group_keys=False).apply(
            lambda x: x.sample(frac=args.downsampling_ratio)
        )
        data_name = f"train_v1_r_{args.downsampling_ratio}.jsonl"

    train_codes = list(train["code"].unique())
    all_codes_new = set(dev_codes).union(set(test_codes)).union(set(train_codes))
    assert all_codes == all_codes_new

    train.to_json(
        os.path.join(args.train_aug, data_name),
        lines=True,
        orient="records",
        force_ascii=False,
    )


if __name__ == "__main__":
    args = pars_args()
    main(args)

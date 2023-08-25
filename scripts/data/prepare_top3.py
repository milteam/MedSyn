"""Managing RuMedTop3 data and upsampling with templates."""

import os
import random
import json
import argparse
import numpy as np
import pandas as pd


FULL_TRAIN_SET = 4690


def pars_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--downsampling-ratio",
        default=1,
    )
    argparser.add_argument(
        "--ratio",
        default=0.75,
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
        default="data/benchmarks/RuMedTop3",
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

        # Constrained sampling:
        samples_df = pd.DataFrame(columns=["icd_full", "icd", "symptoms"])
        for idx, sample in enumerate(common_samples):
            pre = np.random.choice(["", "Жалуется на ", "Пациент обратился с "])
            idx = "{:08d}".format(idx)
            symptoms = pre + ", ".join(sample["symptoms"])
            code_full = sample["disease"][0]["icd_code"]
            code = code_full.split(".")[0]
            samples_df.loc[idx] = [
                code_full,
                code,
                symptoms,
            ]

        frac = args.ratio * FULL_TRAIN_SET / samples_df.shape[0]
        samples_df_sampled = samples_df.groupby("icd", group_keys=False).apply(
            lambda x: x.sample(frac=frac, replace=False)
        )

        pad = len(train)
        for idx in range(len(samples_df_sampled)):
            sample = samples_df_sampled.iloc[idx]
            idx_ = "{:08d}".format(idx)
            data = [idx_, sample["symptoms"], sample["icd"]]
            train.loc[pad + idx] = data

        data_name = f"train_v1_aug_{args.ratio}_cons_temp.jsonl"

    else:
        train = train.groupby("code", group_keys=False).apply(
            lambda x: x.sample(frac=args.downsampling_ratio, replace=False)
        )
        data_name = f"train_v1_dsr_{args.downsampling_ratio}.jsonl"

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
import json
import os
import random

import fire
from tqdm import tqdm

from utils import get_filepaths

NON_MED_KEYS = ["sberquad", "ru_facts", "multidomaingold_spellcheck"]


def merge_med_non_med_data(
        results_dir: str = "data/data_ift/merged",
        result_name: str = "all_data_merged_ift.jsonl",
        samples_path: str = "data/data_ift",
        med_data_percent: int = 80,
        num_min_samples: int = 3000,
):
    os.makedirs(results_dir, exist_ok=True)

    filepaths = get_filepaths(samples_path)

    sizes = dict()
    raw_data = dict()
    for filepath in filepaths:
        if filepath.endswith(".jsonl"):
            with open(filepath, encoding="utf-8") as r:
                data = [json.loads(line) for line in r]
                ln = len(data)
                sizes[filepath] = ln
                raw_data[filepath] = data
        else:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
                ln = len(data)
                sizes[filepath] = ln
                raw_data[filepath] = data

    med_data_counter = sum(v for k, v in sizes.items() if not any(nmd for nmd in NON_MED_KEYS if nmd in k))
    non_med_data_counter = sum(v for k, v in sizes.items() if any(nmd for nmd in NON_MED_KEYS if nmd in k))
    print("\nMed data length:", med_data_counter, "\nNon-Med data length:", non_med_data_counter)

    num_non_med_data = round(
        ((100 - med_data_percent) * med_data_counter) / med_data_percent
    )
    print(f'{100 - med_data_percent}% of med data:', num_non_med_data)

    random.seed(42)

    # collect non med data
    all_data = list()
    for nmd in NON_MED_KEYS:
        if "sberquad" in nmd:
            continue

        filepath = [k for k in sizes.keys() if nmd in k][0]
        data = raw_data[filepath]
        random.shuffle(data)
        all_data.extend(data[:num_min_samples])

    # sberquad gets more samples
    sber_filepath = [k for k in sizes.keys() if "sberquad" in k][0]
    data = raw_data[sber_filepath]
    random.shuffle(data)
    all_data.extend(data[: (num_non_med_data - len(all_data))])

    # collect med data
    for filepath in sizes:
        # skip non med data
        if not any(nmd for nmd in NON_MED_KEYS if nmd in filepath):
            data = raw_data[filepath]
            random.shuffle(data)
            all_data.extend(data)
    print(f'All med data + {100 - med_data_percent}% of non-med data:', len(all_data))

    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        for row in tqdm(data):
            w.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(merge_med_non_med_data)

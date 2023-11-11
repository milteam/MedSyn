import os
import json
import pandas as pd
from os.path import join

from tqdm import tqdm


def merge_med_data(
    results_dir: str = "data/data_ift/merged",
    result_name: str = "all_data_merged_ift.jsonl",
    samples_path: str = "data/data_ift",
):
    os.makedirs(results_dir, exist_ok=True)

    merged = []

    for dir in tqdm(os.listdir(samples_path)):
        for file in os.listdir(join(samples_path, dir)):
            f_path = join(samples_path, dir, file)

            print(f_path)

            data = pd.read_json(f_path)
            for idx in data.index:
                instruction = data.loc[idx]["instruction"]
                input = data.loc[idx]["input"]
                output = data.loc[idx]["output"]
                sample = {"instruction": instruction, "input": input, "output": output}
                merged.append(sample)

    print(f"{len(merged)} samples generated.")

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(merged, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    merge_med_data()
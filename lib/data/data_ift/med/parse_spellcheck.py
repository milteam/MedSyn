from typing import List, Dict

import json
import os

import fire
from tqdm import tqdm


def get_filepaths(root_dir: str) -> List[str]:
    filepaths = list()

    for path, subdirs, files in os.walk(root_dir):
        for name in files:
            filepaths.append(os.path.join(path, name))

    return filepaths


def parse(root_dir: str) -> List[Dict]:
    filepaths = get_filepaths(root_dir)

    all_data = list()
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                all_data.append(row)

    return all_data


def generate_data(
    results_dir: str = "data/data_ift/spellcheck_benchmark",
    result_name: str = "spellcheck_data_ift.json",
    samples_path: str = "data/data_raw/spellcheck_benchmark",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    prompt = "Ты хорошо знаешь русский язык. Исправь орфографические и грамматические ошибки в тексте."

    filepaths = get_filepaths(samples_path)

    for filepath in filepaths:
        filename = filepath.split(r"/")[-2]
        data = parse(os.path.join(samples_path, filename))

        result = []

        for row in tqdm(data):
            instruction = {
                "instruction": prompt,
                "input": row["source"],
                "output": row["correction"],
            }
            result.append(instruction)

        with open(
            os.path.join(results_dir, f"{filename.lower()}_{result_name}"),
            "w",
            encoding="utf8",
        ) as f:
            json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(generate_data)

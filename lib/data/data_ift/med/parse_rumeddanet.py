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
    results_dir: str = "data/data_ift/rumeddanet",
    result_name: str = "rumeddanet_data_ift.json",
    samples_path: str = "data/data_raw/RuMedDaNet",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    data = parse(samples_path)

    result = []

    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        prompt = """Ты являешься профессиональным врачом. Учитывая контекст, ответь на вопрос максимально точно. В ответе обязательно напиши один из вариантов "да" или "нет"."""
        inp = "Контекст: {0}\nВопрос: {1}"
        for row in tqdm(data):
            if row.get("answer", None) is None:
                continue
            instruction = {
                "instruction": prompt,
                "input": inp.format(row["context"], row["question"]),
                "output": f'Ответ: {row["answer"]}.',
            }
            result.append(instruction)

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(generate_data)

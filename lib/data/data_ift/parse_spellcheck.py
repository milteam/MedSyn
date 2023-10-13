import json
import os

import fire
from tqdm import tqdm

from utils import parse


def generate_data(
        results_dir: str = "data/data_ift/spellcheck_benchmark",
        result_name: str = "spellcheck_data_ift.jsonl",
        samples_path: str = "data/data_raw/spellcheck_benchmark",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    data = parse(samples_path)

    prompt = "Ты хорошо знаешь русский язык. Исправь орфографические и грамматические ошибки в тексте."
    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        for row in tqdm(data):
            instruction = {
                "instruction": prompt,
                "input": row["source"],
                "output": row["correction"],
            }
            w.write(json.dumps(instruction, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(generate_data)

import json
import os

import fire
from tqdm import tqdm

from utils import parse


def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        return [json.loads(line) for line in r]


def generate_data(
        results_dir: str = "data/data_ift/ru_facts",
        result_name: str = "ru_facts_data_ift.jsonl",
        samples_path: str = "data/data_raw/ru_facts",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    data = parse(samples_path)

    prompt = "Учитывая посылку: '{0}' и гипотезу: '{1}', определи тип связи между этими двумя предложениями. Предложения являются противоречивыми или непротиворечивыми?"
    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        for row in tqdm(data):
            instruction = {
                "instruction": prompt.format(row["evidence"], row["claim"]),
                "output": f"Ответ: предложения {'противоречивые' if row['label']==1 else 'непротиворечивые'}.",
            }
            w.write(json.dumps(instruction, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(generate_data)

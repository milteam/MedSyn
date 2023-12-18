import json
import os
from typing import List, Dict

import fire
from tqdm import tqdm

from lib.data.data_ift.utils.utils import get_filepaths


def parse(root_dir: str) -> List[Dict]:
    filepaths = get_filepaths(root_dir)

    all_data = list()
    for filepath in filepaths:
        with open(filepath, encoding="utf-8") as f:
            squad = json.load(f)
            for article in squad["data"]:
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"]
                    for qa in paragraph["qas"]:
                        answer = [answer["text"] for answer in qa["answers"]][0]
                        row = {
                            "context": context,
                            "question": qa["question"],
                            "id": qa["id"],
                            "answer_text": answer,
                        }
                        all_data.append(row)
    return all_data


def generate_data(
        results_dir: str = "data/data_ift/sberquad",
        result_name: str = "sberquad_data_ift.jsonl",
        samples_path: str = "data/data_raw/sberquad",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    data = parse(samples_path)

    prompt = "Учитывая контекст, дай максимально точный ответ на вопрос."
    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        for row in tqdm(data):
            instruction = {
                "instruction": prompt,
                "input": f'Контекст: {row["context"]}\nВопрос:{row["question"]}',
                "output": f'Ответ: {row["answer_text"]}.',
            }
            w.write(json.dumps(instruction, ensure_ascii=False, indent=3) + "\n")


if __name__ == "__main__":
    fire.Fire(generate_data)

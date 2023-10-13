import json
import os

import fire
from tqdm import tqdm

from utils import parse


def generate_data(
        results_dir: str = "data/data_ift/rumeddanet",
        result_name: str = "rumeddanet_data_ift.jsonl",
        samples_path: str = "data/data_raw/RuMedDaNet",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    data = parse(samples_path)

    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        prompt = """Ты являешься профессиональным врачом. Учитывая контекст, ответь на вопрос максимально точно.
Контекст: {0}
Вопрос: {1}
В ответе обязательно напиши один из вариантов "да" или "нет"."""
        for row in tqdm(data):
            if row.get("answer", None) is None:
                continue
            instruction = {
                "instruction": prompt.format(row["context"], row["question"]),
                "output": f'Ответ: {row["answer"]}.',
            }
            w.write(json.dumps(instruction, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(generate_data)

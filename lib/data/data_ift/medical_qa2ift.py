"""Convert medical_qa_ru data to IFT format."""

from typing import Dict, Set, List

import os
import re
import random
from tqdm import tqdm
import pandas as pd
import json
import click

INSTRUCTIONS = [
    "Напиши ответ",
    "Ты являешься врачом, напиши ответ по воросу пациента",
    "Ты врач, ответь пациенту",
    "Дай медицински корректный ответ",
    "Напиши врачебную консультацию",
    "Напиши медицинский ответ",
]

STOP_WOERS = [
    "спасибо Вам;",
    "Спасибо",
    "cпасибо",
    "Будьте здоровы!",
    "Здравствуйте",
    "Здравствуйте!",
    "Не болейте!",
    "Доброй ночи!",
    "Добрый вечер",
    "Добрый день",
    "добрый день!",
    "Доброе утро!",
    "Доброе утро",
    "День добрый",
    "ЗДРАВСТВУЙТЕ!",
    "!",
    "!!",
    "!!!",
    ";",
]


def remove_words(text):
    pattern = r"\b(?:" + "|".join(re.escape(word) for word in STOP_WOERS) + r")\b"
    cleaned_text = re.sub(pattern, " ", text)
    text = " ".join(cleaned_text.split())
    text = re.sub(r" +", " ", text).replace(";", "")
    text = re.sub(r"\s+([.,])", r"\1", text)
    text = re.sub(r"^[\W_]+", "", text).strip()
    return text


def get_sample(question: str, answer: str) -> Dict:
    instruction = random.choice(INSTRUCTIONS)
    sample = {"instruction": instruction, "input": question, "output": answer}
    return sample


def preprocess(
    data: pd.DataFrame, min_words: int = 10, max_words: int = 30
) -> pd.DataFrame:
    # Strings only:
    str_mask = data["ans"].apply(lambda x: type(x) == str)
    data = data[str_mask]

    str_mask = data["desc"].apply(lambda x: type(x) == str)
    data = data[str_mask]

    # Remove redundent words:
    data["ans"] = data["ans"].apply(lambda x: remove_words(x))
    data["desc"] = data["desc"].apply(lambda x: remove_words(x))

    q_mask = data["ans"].apply(lambda x: "?" not in x)
    data = data[q_mask]

    q_mask = data["ans"].apply(lambda x: "!" not in x)
    data = data[q_mask]

    min_l_mask = data["ans"].apply(lambda x: len(x.split(" ")) > min_words)
    data = data[min_l_mask]

    min_l_mask = data["desc"].apply(lambda x: len(x.split(" ")) > min_words)
    data = data[min_l_mask]

    max_l_mask = data["ans"].apply(lambda x: len(x.split(" ")) < max_words)
    data = data[max_l_mask]

    max_l_mask = data["desc"].apply(lambda x: len(x.split(" ")) < max_words)
    data = data[max_l_mask]

    return data


@click.command()
@click.option("--results-dir", default="data/data_ift/medical_qa")
@click.option("--result-name", default="medical_qa.json")
@click.option("--data-path", default="data/data_raw/medical_qa_ru_data.csv")
def generate_data(
    results_dir: str,
    result_name: str,
    data_path: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []
    data = pd.read_csv(data_path)

    data = preprocess(data)

    for idx in tqdm(data.index):
        question = data.loc[idx]["desc"]
        answer = data.loc[idx]["ans"]

        new_sample = get_sample(question, answer)
        result.append(new_sample)

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

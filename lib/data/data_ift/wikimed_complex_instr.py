import json
import os
import pandas as pd
import random

from tqdm import tqdm
from transformers import AutoTokenizer

QUESTIONS_DRUG = {
    "Противопоказания": "Какие противопоказания у вещества",
    "Побочные действия": "Какие побочные действия у вещества",
    "Взаимодействие": "Какое взаимодействие вещества с другими",
    "Способ применения и дозы": "Какой способ применения и дозы у вещества",
    "Меры предосторожности": "Какие меры предосторожности у вещества",
    "МКБ-10": "Какие есть заболевания, для которых может применяться вещество",
}

QUESTIONS_DIS = {
    "Этиология и патогенез": "Какой патогенез заболевания",
    "Клинические проявления": "Какие клинические проявления заболевания",
    "Диагностика": "Как диагностировать заболевание",
    "Дифференциальный диагноз": "Какой дифференциальный диагноз у заболевания",
    "Лечение": "Какое лечение у заболевания",
    "Профилактика": "Какая профилактика у заболевания",
    "Действующие вещества": "Какие действующие вещества нужны для лечения заболевания",
}

random.seed(42)

base_model = "IlyaGusev/gigasaiga_lora"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)


def generate_data_for_type(
        wikimed_path: str, results_dir: str, result_name: str,
        questions: dict, max_tokens_thresh: int = 2000):
    os.makedirs(results_dir, exist_ok=True)

    wikimed = pd.read_csv(wikimed_path)

    result = list()
    instr_text = f"Максимально подробно опиши {'заболевание' if 'diseases' in result_name else 'лекарственное средство'}."
    for idx in wikimed.index:
        row = wikimed.loc[idx]
        name = row['Рубрика'] if 'diseases' in result_name else row['Название']

        description = []
        for instruction_type in questions.keys():
            if row[[instruction_type]].notna().bool():
                description.append(
                    f"{questions[instruction_type]} {name}? {row[instruction_type]}.")
        if len(description) <= 1:
            continue
        random.shuffle(description)

        instruction = {
            "instruction": instr_text,
            "input": name,
            "output": description,
        }
        result.append(instruction)

    for instr in result:
        concat_output = ''
        for i in instr['output']:
            i = i.replace("\n", "").replace("\r", "")
            tokens = tokenizer(
                concat_output + f'{i} ',
                return_tensors="pt",
            )
            num_tokens = len(tokens['input_ids'][0])
            if num_tokens < max_tokens_thresh:
                concat_output += f'{i} '
        instr['output'] = concat_output

    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        for row in tqdm(result):
            if len(row['output']) == 0:
                continue
            w.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_data(
        results_dir: str = "data/data_ift/wikimed_complex",
        wikimed_drugs_path: str = "data/data_raw/wikimed/wikimed_meds.csv",
        wikimed_dis_path: str = "data/data_raw/wikimed/wikimed_diseases.csv",
        max_tokens_thresh: int = 2000
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    generate_data_for_type(wikimed_drugs_path, results_dir, result_name='drugs.jsonl',
                           questions=QUESTIONS_DRUG, max_tokens_thresh=max_tokens_thresh)
    generate_data_for_type(wikimed_dis_path, results_dir, result_name='diseases.jsonl',
                           questions=QUESTIONS_DIS, max_tokens_thresh=max_tokens_thresh)

"""Convert RuMedNLI data to IFT data format."""

from typing import Dict, Set

import os
import json
import click
import random
import pandas as pd
from tqdm import tqdm

instruction_ent = [
    "Напиши медицинское утверждение, которое следует из исходного.",
    "Сформулируй медицинское заключение, основанное на предоставленной информации.",
    "Вырази медицинское суждение, вытекающее из первоначального утверждения.",
    "Опиши вывод, который можно сделать на основе данного медицинского утверждения.",
    "Произведи медицинское заключение, соответствующее исходному утверждению.",
    "Создай медицинское утверждение, являющееся логическим продолжением первоначального.",
]
instruction_cont = [
    "Напиши медицинское утверждение, которое противоречит исходному.",
    "Сформулируй медицинское заключение, контрастирующее с первоначальным утверждением.",
    "Опиши медицинское суждение, которое находится в оппозиции к исходному.",
    "Вырази медицинское мнение, противоположное данному утверждению.",
    "Произведи медицинское утверждение, которое является антитезисом первоначальному.",
    "Создай медицинское заявление, диаметрально противопоставленное исходному.",
]
instruction_neut = [
    "Напиши медицинское утверждение, для которого нельзя сказать что оно не противоречит исходному, также нельзя сказать, что оно явлется его следствием.",
    "Сформулируй медицинское заключение, которое не является ни противоречием, ни следствием исходного утверждения.",
    "Опиши медицинское утверждение, не связанное напрямую с исходным, ни как противоречие, ни как следствие.",
    "Вырази медицинское мнение, которое не может быть классифицировано как прямое противоречие или логическое следствие исходного.",
    "Произведи медицинское заявление, стоящее вне рамок прямого противоречия или следствия по отношению к первоначальному утверждению.",
    "Создайте медицинское суждение, которое не является ни отрицанием, ни продолжением исходного утверждения.",
]


@click.command()
@click.option("--result-name", default="rumednli.json")
@click.option("--nli-root", default="data/data_raw/RuMedNLI")
@click.option("--results-dir", default="data/data_ift/rumednli")
def generate_data(nli_root: str, result_name: str, results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)

    result = []

    for data in tqdm(os.listdir(nli_root)):
        file = os.path.join(nli_root, data)

        records = pd.read_json(file, lines=True)

        for idx in records.index:
            input = records.loc[idx]["ru_sentence1"]
            output = records.loc[idx]["ru_sentence2"]
            label = records.loc[idx]["gold_label"]

            if label == "entailment":
                instruction = random.choice(instruction_ent)
                sample = {"instruction": instruction, "input": input, "output": output}
            if label == "contradiction":
                instruction = random.choice(instruction_cont)
                sample = {"instruction": instruction, "input": input, "output": output}
            if label == "neutral":
                instruction = random.choice(instruction_neut)
                sample = {"instruction": instruction, "input": input, "output": output}

            result.append(sample)

    print(f"{len(result)} samples generated.")

    with open(os.path.join(results_dir, result_name), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

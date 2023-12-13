import click
import concurrent.futures
import datetime
import json
import openai
import os
import numpy as np
import pandas as pd
from functools import partial
from glob import glob
from openai import OpenAIError
from pathlib import Path
from typing import Tuple


SAMPLE_ID = str
PROMPT = str


SYSTEM_INSTRUCTION = """Проанализируй клинические проявления и исходный анамнез.
Ответь, используя только симптомы из запроса.
Воздержись от размышлений о том, что этот анамнез для другого пациента и является корректным.
Избегай общих описаний заболевания.
Дай краткий ответ, прямо связанный с анамнезом.
Твой ответ не должен содержать противоречий.
Следуй стилю исходного анамнеза без дублирования информации.
Не задавай уточняющих вопросов и не добавляй примечаний.
Избегай приветствий, комментариев и упоминаний врача.
Ответ должен быть на русском языке."""

PROMPT_STRUCTURE = """Представь, что являешься врачем с высоким уровнем владения русским языком.

Внимательно рассмотри исходный анамнез пациента:
{anamnesis}

Напиши аналогичный анамнез примерно же объема, как и исходный анамнез, для другого пациента с таким же заболеванием,
в котором обязательно замени часть симптомов на перечисленные далее симптомы, используя их всех:
{symptoms}.

Убедись, что этот анамнез отличается от исходного и является корректным с медицинской точки зрения."""


def get_gpt_response(prompt: str, gpt_version: str) -> str:
    response = openai.ChatCompletion.create(
        model=gpt_version,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ],
    )
    return response["choices"][0]["message"]["content"]


def submit_response(output_folder: Path, sample_id: int, response: str, prompt: str):
    filename = f"{sample_id}.json"
    with open(output_folder / filename, "wt", encoding="utf-8") as file:
        json.dump(dict(
            sample_id=sample_id,
            response=response,
            instruction=SYSTEM_INSTRUCTION,
            prompt=prompt,
            created=str(datetime.datetime.now())
        ), file, ensure_ascii=False)


def generate_anamnesis(params: Tuple[SAMPLE_ID, PROMPT], output_folder: str, gpt_version: str):
    for sample_id, prompt in params:
        if len(prompt) > 6000:
            print(f"\033[31mToo long prompt for {sample_id}.\033[0m")
            return
        try:
            response = get_gpt_response(prompt, gpt_version)
            submit_response(output_folder, sample_id, response, prompt)
        except OpenAIError as ex:
            print(f"OpenAI Exception for sample_id = {sample_id}: \033[31m{ex}\033[0m")
        except Exception as ex:
            print(f"Exception for sample_id = {sample_id}: \033[31m{ex}\033[0m")


@click.command()
@click.option("--taskgen-path", required=True)
@click.option("--output-folder", required=True)
@click.option("--gpt-version", default="gpt-4")
def main(taskgen_path: str, output_folder: str, gpt_version: str):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    if openai.api_key is None:
        print("\033[31mThere is no OpenAI API key.\033[0m")
        exit(1)

    taskgen_df = pd.read_csv(taskgen_path).rename(columns={"shuffled_symptoms": "symptoms"})
    taskgen_df.set_index(taskgen_df["sample_id"], inplace=True)

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    handled_samples = [Path(e).stem for e in glob(os.path.join(output_folder, "*.json"))]
    taskgen_df = taskgen_df[~taskgen_df["sample_id"].isin(handled_samples)]

    sample_params = []
    for sample_id in taskgen_df["sample_id"]:
        d = taskgen_df.loc[sample_id, ["anamnesis", "symptoms"]].to_dict()
        d["symptoms"] = d["symptoms"].replace("|", "; ")
        prompt = PROMPT_STRUCTURE.format(**d)
        sample_params.append((sample_id, prompt))

    handler = partial(generate_anamnesis, output_folder=output_folder, gpt_version=gpt_version)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        r = list(executor.map(handler, np.array_split(sample_params, 10)))


if __name__ == "__main__":
    main()
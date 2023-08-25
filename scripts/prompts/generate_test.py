import json
import os
import random
import re
import time

import click
import openai
from openai.error import OpenAIError, ServiceUnavailableError
from scripts.prompts.russian import get_russian_details

INSTRUCTION = "Напиши текст анамнеза, составленного врачом по итогам приема пациента."


def get_alpaca_prompt(sample):
    symptoms = sample["symptoms"]
    gender = sample["gender"]
    marital_state = sample["family_state"]
    smoking = sample["smoking"]

    conj, gender_ru, marital, smoking = get_russian_details(
        gender, marital_state, smoking
    )
    symptoms = ", ".join(symptoms).lower()
    input = f"Пациент - {marital} {smoking} {gender_ru}, {conj} жалуется на {symptoms}."

    output = ""

    sample = {"instruction": INSTRUCTION, "input": input, "output": output}
    return sample


# На момент написания анамнеза диагноз пациента не известен и не должен быть упомянут в ответе.


@click.command()
@click.option("--output", "-o", help="Output file name", required=True)
@click.option("--samples", "-s", help="JSON files with deseases", required=True)
def main(output, samples):
    random.seed(0)

    res = []
    with open(samples, encoding="utf8") as f:
        data = json.load(f)

        for item in data.values():
            res.append(get_alpaca_prompt(item))

    with open(output, "w", encoding="utf8") as f:
        json.dump(res, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    main()

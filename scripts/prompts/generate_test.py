import json
import os
import random
import re
import time
import openai
import click
from openai.error import ServiceUnavailableError, OpenAIError

INSTRUCTION = "Напиши текст анамнеза, составленного врачом по итогам приема пациента."


def get_sample(data):
    uid = data["UID"]
    desease_code = data["disease"][0]["idc_10"]
    symptoms = data["symptoms"]
    # age = data["age"]
    gender = data["gender"]
    marital_state = data["family_state"]
    smoking = data["smoking"]

    desaese_name = str(data["disease"][0]["name_ru"]).lower()
    if gender == "male":
        marital = "женатый" if marital_state else "неженатый"
        smoking = "курящий" if smoking else "некурящий"
        gender_ru = "мужчина"
        conj = "который"
    else:
        marital = "замужняя" if marital_state else "незамужняя"
        smoking = "курящая" if smoking else "некурящая"
        gender_ru = "женщина"
        conj = "которая"
        # Известно, что пациент - {marital} {smoking} {gender_ru}.
    # if len(symptoms) > 3:
    #     symptoms = random.sample(symptoms, random.randint(1,4))
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
            res.append(get_sample(item))


    with open(output, "w", encoding="utf8") as f:
        json.dump(res, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    main()

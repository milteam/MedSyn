import json
import os
import random
import openai

import click
from tqdm import tqdm

from graph import sample_disease
from prompt_generator import generate_prompt

def generate_prompt_ru(data):
    actual_desease = data["disease"][0]["Name"]
    symptoms = data["symptoms"]
    if len(symptoms) > 3:
        symptoms = random.sample(symptoms, random.randint(1,4))
    return actual_desease, symptoms, f'''
Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный отчет.
В действительности диагноз пациента - {actual_desease}. Но на момент написания анамнеза это еще неизвестно.
# Пациент жалуется на {", ".join(symptoms)}. Реалистично придумай все недостающие данные для отчета.'''

@click.command()
@click.option('--out', '-o',
              help='Output file name', required=True)
@click.option('--samples', '-s', help='JSON files witj deseases', required=True)
def main(out, samples):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    res = []
    with open(samples, encoding="utf8") as f:
        data = json.load(f)
        for idx, desease in data.items():
            actual_desease, symptoms, prompt = generate_prompt_ru(desease)
            response = openai.ChatCompletion.create(model="gpt-4-0314", messages=[{"role": "user", "content": prompt}])
            text = response['choices'][0]['message']['content']
            tqdm.write(text)
            print("========================")
            res.append({'desease': actual_desease, 'symptoms': symptoms, 'description': text})


    with open(out, 'w', encoding='utf8')  as f:
        json.dump(res, f, indent=3, ensure_ascii=False)

if __name__ == '__main__':
    main()
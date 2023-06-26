import json
import random

import click

from graph import DiseaseInfo



def generate_prompt_ru(info: DiseaseInfo):
    return [{"role": "user", "content": f'''
            Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный отчет.
Представь, что у пациента есть слеующие хронические заболевания: {", ".join(info.preconditions)}.
В действительности диагноз пациента - {info.name}. Но на момент написания анамнеза это еще неизвестно.
Пациент жалуется на {", ".join(info.symptoms)}. Реалистично придумай все недостающие данные для отчета.'''},
            ]

def generate_prompt_en(data):
    actual_desease = data["disease"][0]["Name"]
    symptoms = data["symptoms"]
    if len(symptoms) > 3:
        symptoms = random.sample(symptoms, random.randint(1,4))
    return f'''
    Act like a professional doctor who listened to the patient. Write a well-structured and extremely detailed medical anamnesis that includes all the required sections.        
  In fact, the patient has {actual_desease}, but the doctor does not know about it.
  The patient complains of {", ".join(symptoms)}.
  '''

@click.command()
@click.option('--samples', '-s', help='JSON files witj deseases', required=True)
def main(samples):
    with open(samples, encoding="utf8") as f:
        data = json.load(f)
        for idx, desease in data.items():
            print(generate_prompt_en(desease))

if __name__ == '__main__':
    main()
import json
import os
import random
import time
import openai
import click
from openai.error import ServiceUnavailableError, OpenAIError


def prompt_no_patent_info(symptoms, diagnose, marital, smoking, gender, conj ):
    return f'''
    Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный анамнез.
    Пациент жалуется на {symptoms}.
    Реалистично придумай все недостающие детали, характеристики пациента и сопутствующие симпотмы для анамнеза. '''


def prompt_with_patient_info(symptoms, diagnose, marital, smoking, gender, conj ):
    return f'''
Напиши текст анамнеза, составленного врачом по итогам приема пациента.
Пациент - {marital} {smoking} {gender}, {conj} жалуется на {symptoms}.
Придумай все недостающие детали, характеристики пациента и сопутствующие симпотмы для анамнеза. '''


def short_promt(symptoms, diagnose, marital, smoking, gender, conj ):
    return f'''
Представь, что являешься врачом. Напиши краткий анамнез по итогам приема пациента, который жалуется на {symptoms}.
 Дополни все недостояющая детали реалистичными данными. '''

def prompt_diagnostics(symptoms, diagnose, marital, smoking, gender, conj ):
    return f'''
    Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный анамнез.
    Пациент жалуется на {symptoms}.
    Реалистично придумай все недостающие детали и результаты исследований, которые могли быть назначены в такой ситуации. '''

PROMPTS = [prompt_no_patent_info, prompt_with_patient_info, short_promt, prompt_diagnostics]

def get_sample(data):
    desease_code = data["disease"][0]["idc_10"]
    symptoms = data["symptoms"]
    #age = data["age"]
    gender = data["gender"]
    marital_state = data["family_state"]
    smoking = data["smoking"]

    desaese_name = (data["disease"][0]["name_ru"] or "").lower()
    if gender == "male":
        marital = "замужний" if marital_state else "незамужний"
        smoking = "курящий" if smoking else "некурящий"
        gender_ru = "мужчина"
        conj = "который"
    else:
        marital = "замужняя" if marital_state else "незамужняя"
        smoking = "курящая" if smoking else "некурящая"
        gender_ru = "женщина"
        conj = "которая"
        #Известно, что пациент - {marital} {smoking} {gender_ru}.
    # if len(symptoms) > 3:
    #     symptoms = random.sample(symptoms, random.randint(1,4))
    return dict(desease_code=desease_code, symptoms=symptoms, gender=gender, marital_state=marital_state, smoking=smoking,
                desease_name=desaese_name,
                prompt=random.choice(PROMPTS)(", ".join(symptoms).lower(), desaese_name,
                                              marital, smoking, gender_ru, conj))
#На момент написания анамнеза диагноз пациента не известен и не должен быть упомянут в ответе.


@click.command()
@click.option('--dir', '-d',
              help='Output dir name', required=True)
@click.option('--samples', '-s', help='JSON files with deseases', required=True)
@click.option('--limit', '-l', help='A number of sentences to generate', default=100)
@click.option('--offset', '-o', help='A number of sentences to skip', default=0)
def main(dir, samples, limit, offset):
    random.seed(0)

    openai.api_key = os.getenv("OPENAI_API_KEY")

    tic = time.perf_counter()

    if not os.path.exists(dir):
        os.makedirs(dir)
    res = []
    with open(samples, encoding="utf8") as f:
        data = json.load(f)
        print(f"Generating {limit} samples out of {len(data)} starting from {offset}")

        data = list(data.values())
        idx = offset
        while idx < len(data):
            try:
                if idx % 20 == 0 and idx > offset:
                    print(f"Storing results from {idx-20} to {idx}")
                    with open(os.path.join(dir, f"result_{idx-20}-{idx}.json"), 'w', encoding='utf8') as f:
                        json.dump(res, f, indent=3, ensure_ascii=False)
                        res = []

                print(f"\n\n\n======================== Sample {idx+1} ========================")
                desease_info = get_sample(data[idx])
                print(desease_info["desease_name"])
                print("\n\n")
                print(desease_info["prompt"])
                print("-----------------------------------------------------------------")
                response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": desease_info["prompt"]}])
                desease_info["response"] = response['choices'][0]['message']['content']
                print(desease_info["response"])
                res.append(desease_info)
                idx += 1
                if idx-offset == limit:
                    break
            except OpenAIError as e:
                print(f"OpeanAI Exception, service is busy, waiting a few seconds: {e}")
                time.sleep(5)
                continue


    toc = time.perf_counter()
    print(f"Time to generate {limit} samples {toc-tic} seconds")

    with open(os.path.join(dir, f"result_{idx - 20}-{idx}.json"), 'w', encoding='utf8') as f:
        json.dump(res, f, indent=3, ensure_ascii=False)

if __name__ == '__main__':
    main()
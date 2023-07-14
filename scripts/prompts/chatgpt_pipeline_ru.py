import json
import os
import time
import openai
import click


def get_sample(data):
    desease_code = data["disease"][0]["idc_10"]
    symptoms = data["symptoms"]
    #age = data["age"]
    gender = data["gender"]
    marital_state = data["family_state"]
    smoking = data["smoking"]
    if gender == "male":
        marital = "замужний" if marital_state else "незамужний"
        smoking = "курящий" if smoking else "некурящий"
        gender_ru = "мужчина"
    else:
        marital = "замужняя" if marital_state else "незамужняя"
        smoking = "курящая" if smoking else "некурящая"
        gender_ru = "женщина"
        #Известно, что пациент - {marital} {smoking} {gender_ru}.
    # if len(symptoms) > 3:
    #     symptoms = random.sample(symptoms, random.randint(1,4))
    return dict(desease_code=desease_code, symptoms=symptoms, gender=gender, marital_state=marital_state, smoking=smoking,
                prompt=f'''
Напиши ответ как будто являешься профессиональным врачом, который записывает анамнез со слов пациента. Напиши хорошо структурированный и детальный отчет.
Пациент жалуется на {", ".join(symptoms).lower()}. 
Реалистично придумай все недостающие данные для отчета как если бы у пациента был {data["disease"][0]["name_ru"].lower()}. 
Диагноз не должен быть написан в анамнезе.
Фраза "{data["disease"][0]["name_ru"].lower()}" не должна упоминаться в ответе ни в каком виде.''')
#На момент написания анамнеза диагноз пациента не известен и не должен быть упомянут в ответе.


@click.command()
@click.option('--out', '-o',
              help='Output file name', required=True)
@click.option('--samples', '-s', help='JSON files with deseases', required=True)
@click.option('--limit', '-l', help='A number of sentences to generate', default=5)

def main(out, samples, limit):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    tic = time.perf_counter()

    res = []
    with open(samples, encoding="utf8") as f:
        data = json.load(f)
        print(f"Generating {limit} samples out of {len(data)}")

        for idx, desease in zip(range(limit), data.values()):
            print(f"\n\n\n======================== Sample {idx+1} ========================")
            desease_info = get_sample(desease)
            print(desease_info["prompt"])
            print("-----------------------------------------------------------------")
            response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": desease_info["prompt"]}])
            desease_info["response"] = response['choices'][0]['message']['content']
            print(desease_info["response"])
            res.append(desease_info)

    toc = time.perf_counter()
    print(f"Time to generate {limit} samples {toc-tic} seconds")

    with open(out, 'w', encoding='utf8')  as f:
        json.dump(res, f, indent=3, ensure_ascii=False)

if __name__ == '__main__':
    main()
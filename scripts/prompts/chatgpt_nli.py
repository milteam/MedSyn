import json
import os
import random
import time

import click
import openai
from nltk import sent_tokenize
from openai import OpenAIError

from scripts.prompts import english, russian
from scripts.prompts.utils import (MedicalRecord, generate_gpt_records,
                                   get_gpt_response)

LANG_TAMPLES = {"en": english.get_desase_tamplate, "ru": russian.get_desase_tamplate}


# На момент написания анамнеза диагноз пациента не известен и не должен быть упомянут в ответе.

def contradiction(sentence, gpt):
    prompt = f'''
        Напиши предложение на медицинскую тематику, противоречащее по смыслу данному ниже, не повторяя слова из исходного предложения.
    Предложение - {sentence}
        '''
    return get_gpt_response(prompt, gpt)


def entaiment(sentence, gpt):
    prompt = f'''
        Напиши  предложение на медицинскую тематику, логически следующее из данного ниже, не повторяя слова из исходного предложения.
    Предложение - {sentence}
        '''
    return get_gpt_response(prompt, gpt)


def neutral(sentence, gpt):
    prompt = f'''
        Напиши предложение на медицинскую тематику, логически не связанное с данным ниже, не повторяя слова из исходного предложения.
    Предложение - {sentence}
        '''
    return get_gpt_response(prompt, gpt)

@click.command()
@click.option("--dir", "-d", help="Output dir name", required=True)
@click.option("--samples", "-s", help="JSON files with deseases", required=True)
@click.option("--limit", "-m", help="A number of sentences to generate", default=1)
@click.option("--offset", "-o", help="A number of sentences to skip", default=0)
@click.option(
    "--lang", "-l", help="Language of generation, en and ru are supported", default="ru"
)
@click.option(
    "--gpt",
    "-g",
    help="Version of ChatGPT, GPT-3.5 Turbo by default",
    default="gpt-4",
)
def main(dir, samples, limit, offset, gpt, lang):
    random.seed(0)
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # prompt = '''
    #     Мы собираем датасет для обучения нейронной сети на задаче NLI по медицинской тематике.
    #     Напиши предложение, логически следующее из данного ниже, не повторяя слова из исходного предложения,
    #     которое можно использовать в обучающей выборке.
    # Предложение - Страдает от гипервентиляции, мышечных судорог, паралича и диплопии (двоения в глазах).
    #     ''''
    used = {'data/gpt4_1000/result_80-100.json', 'data/gpt4_1000/result_100-120.json',
             'data/gpt4_1000/result_120-140.json', 'data/gpt4_1000/result_140-160.json',
             'data/gpt4_1000/result_160-180.json'}
    files = [os.path.join('data/gpt_portion', fn) for fn in os.listdir('data/gpt_portion')]
    #files = [f for f in files if f not in used]
    print(files)

    all_res = []

    for file in files:
        print("Processing ", file)
        with open(file) as f:
            source_data = json.load(f)

        for item in source_data:
            sentences = sent_tokenize(item["response"])
            for sentence in sentences:
                res = {"original": sentence}
                try:

                    res["entailment"] = entaiment(sentence, gpt)
                    res["contradiction"] = contradiction(sentence, gpt)
                    res["neutral"] = neutral(sentence, gpt)
                except OpenAIError as e:
                    print(f"OpeanAI Exception, service is busy, waiting a few seconds: {e}")
                    time.sleep(5)
                    continue
                print("Original:  ", res["original"])
                print("contradiction:  ", res["contradiction"])
                print("entailment:  ", res["entailment"])
                print("neutral:  ", res["neutral"])
                print("=====================\n\n")
                all_res.append(res)
                with open("nli_results_3.json", "w") as f:
                    json.dump(all_res, f, indent=3, ensure_ascii=False)

                if len(all_res) % 100 == 0:
                    print(f"Generated {len(all_res)}")



if __name__ == "__main__":
    main()

import json
import os
import random
import openai
import click

from scripts.prompts import english, russian
from scripts.prompts.utils import get_gpt_response, MedicalRecord, generate_gpt_records





LANG_TAMPLES = {"en": english.get_desase_tamplate, "ru": russian.get_desase_tamplate}


# На момент написания анамнеза диагноз пациента не известен и не должен быть упомянут в ответе.


@click.command()
@click.option("--dir", "-d", help="Output dir name", required=True)
@click.option("--samples", "-s", help="JSON files with deseases", required=True)
@click.option("--limit", "-m", help="A number of sentences to generate", default=500)
@click.option("--offset", "-o", help="A number of sentences to skip", default=0)
@click.option("--lang", "-l", help="Language of generation, en and ru are supported", default="ru")
@click.option("--gpt", "-g", help="Version of ChatGPT, GPT-3.5 Turbo by default", default="gpt-3.5-turbo")
def main(dir, samples, limit, offset, gpt, lang):
    random.seed(0)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    generate_gpt_records(samples, offset, limit, dir, gpt, LANG_TAMPLES[lang])




if __name__ == "__main__":
    main()

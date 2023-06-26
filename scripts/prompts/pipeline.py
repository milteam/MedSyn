import json
import os

import click
import openai
from tqdm import tqdm

from graph import sample_disease
from prompt_generator import generate_prompt


@click.command()
@click.option('--out', '-o',
              help='Output file name', required=True)
@click.option('--count', '-c',
              help='A number of samples', required=True, type=int)
def main(out, count):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    res = []
    for _ in tqdm(range(count)):
        info = sample_disease()

        prompt = generate_prompt(info)
        response = openai.ChatCompletion.create(model="gpt-4-0314", messages=prompt)
        text = response['choices'][0]['message']['content']
        tqdm.write(text)
        res.append({'code': info.code, 'symptoms': info.symptoms, 'text': text})


    with open(out, 'w', encoding='utf8')  as f:
        json.dump(res, f, indent=3, ensure_ascii=False)

if __name__ == '__main__':
    main()
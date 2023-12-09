import click
import json
import openai
import os
import pandas as pd
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path
from razdel import sentenize


MAX_LEN = 5000


SYSTEM_INSTRUCTION = """Тщательно проанализируй текст.
Твоя задача выделить симтомы, но не заболевания из текста.
Твой ответ должен содержать только перечень симптомов.
Каждый симптом должен быть в отдельной строке.
Воздержись от размышлений о том, к какому заболеванию могут относиться эти симптомы.
Не задавай уточняющих вопросов.
Не добавляй к ответу примечаний.
Воздержись от любых форм приветствия.
Воздержись от рекомендаций.
Воздержись от комментариев к твоему ответу."""


PROMPT_TEMPLATE = "Выдели из текста симптомы, но не заболевания.\n{text}"


def extract_symptoms(text: str, gpt_version: str):
    prompt = PROMPT_TEMPLATE.format(text=text)
    r = openai.ChatCompletion.create(
        model=gpt_version,
        temperature=0.2,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
    )
    return r["choices"][0]["message"]["content"]


@click.command()
@click.option("--input-filepath", required=True)
@click.option("--output-folder", required=True)
@click.option("--gpt-version", default="gpt-3.5-turbo")
def main(input_filepath: str, output_folder: str, gpt_version: str):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    df = pd.read_csv(input_filepath).reset_index()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    symptoms_from = lambda r: [s.strip() for s in r.split("\n") if s.strip()]

    for e in df.itertuples():
        entry_filepath = output_folder / f"{e.index}.jsonl"
        if not os.path.isfile(entry_filepath):
            with open(entry_filepath, "wt", encoding="utf-8") as file:
                if len(e.text) > MAX_LEN:
                    raw_symptoms = []
                    sentences = [s.text for s in sentenize(e.text)]
                    while sentences:
                        c = list(accumulate([len(s) for s in sentences]))
                        k = bisect_right(c, MAX_LEN)
                        part = " ".join(sentences[:k])
                        response = extract_symptoms(part, gpt_version)
                        raw_symptoms += symptoms_from(response)
                        sentences = sentences[k:]
                else:
                    response = extract_symptoms(e.text, gpt_version)
                    raw_symptoms = symptoms_from(response)

                json.dump(
                    dict(
                        source_index=e.index,
                        ICD10_CAT=e.ICD10_CAT,
                        ICD10_CODE=e.ICD10_CODE,
                        raw_symptoms=raw_symptoms
                    ),
                    file,
                    ensure_ascii=False
                )
                file.write("\n")


if __name__ == "__main__":
    main()
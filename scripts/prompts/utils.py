import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import openai
from openai import OpenAIError


@dataclass
class MedicalRecord:
    desease_code: str
    symptoms: list[str]
    gender: str
    marital_state: bool
    smoking: bool
    desease_name: str
    prompt: str
    response: Optional[str] = None


def get_gpt_response(prompt: str, gpt_version: str) -> str:
    response = openai.ChatCompletion.create(
        model=gpt_version,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["choices"][0]["message"]["content"]


def generate_gpt_records(
    samples_file: str, offset: int, limit: int, dir: str, gpt: str, sampler
):
    tic = time.perf_counter()

    if not os.path.exists(dir):
        os.makedirs(dir)
    res = []
    with open(samples_file, encoding="utf8") as f:
        data = json.load(f)
        print(f"Generating {limit} samples out of {len(data)} starting from {offset}")

        data = list(data.values())
        idx = offset
        while idx < len(data):
            try:
                if idx % 20 == 0 and idx > offset:
                    print(f"Storing results from {idx-20} to {idx}")
                    with open(
                        os.path.join(dir, f"result_{idx-20}-{idx}.json"),
                        "w",
                        encoding="utf8",
                    ) as f:
                        json.dump(res, f, indent=3, ensure_ascii=False)
                        res = []

                print(
                    f"\n\n\n======================== Sample {idx+1} ========================"
                )
                desease_info = sampler(data[idx])
                print(desease_info.desease_name)
                print("\n\n")
                print(desease_info.prompt)
                print(
                    "-----------------------------------------------------------------"
                )

                desease_info.response = get_gpt_response(desease_info.prompt, gpt)
                print(desease_info.response)
                res.append(desease_info)
                idx += 1
                if idx - offset == limit:
                    break
            except OpenAIError as e:
                print(f"OpeanAI Exception, service is busy, waiting a few seconds: {e}")
                time.sleep(5)
                continue

        toc = time.perf_counter()
        print(f"Time to generate {limit} samples {toc - tic} seconds")

        with open(
            os.path.join(dir, f"result_{idx - 20}-{idx}.json"), "w", encoding="utf8"
        ) as f:
            json.dump(res, f, indent=3, ensure_ascii=False)

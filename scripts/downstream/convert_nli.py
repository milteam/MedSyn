import json
import uuid

import pandas as pd

with open("scripts/downstream/data/RuMedNLI/train_ru_llama2_7b_ckpt_3.jsonl", "w") as f:
    with open(
        "scripts/downstream/data/RuMedNLI/ru_llama2_7b_ckpt_3_task_new_params.jsonl"
    ) as source:
        for row in source.readlines():
            row = json.loads(row)
            if row["input"] is None or row["answer"] is None:
                print("Skipping")
                continue
            f.write(
                json.dumps(
                    {
                        "idx": str(uuid.uuid4()),
                        "ru_sentence1": row["input"],
                        "ru_sentence2": row["answer"],
                        "gold_label": row["gold_label"],
                    },
                    ensure_ascii=False,
                )
            )
            f.write("\n")

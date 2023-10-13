import json
import os

import fire
from tqdm import tqdm

FILES_TO_FILTER = {
    'wikimed': [
        ('acting_meds', 0),
        ("diagnostics", 10),
        ("differential", 0),
        ("dgug_cautiuons", 0),
        ("drug_char", 10),
        ("drug_connection", 10),
        ("drug_diseases", 0),
        ("drug_dosage", 10),
        ("drug_neg", 10),
        ("drug_neg_symp", 10),
        ("drug_pharm_group", 30),
        ("drug_pharmacology", 30),
        ("drug_usage", 5),
        ("manifestations", 30),
        ("pathogenesis", 30),
        ("prevention", 30),
        ("treatment", 30),
    ],
    'rumedprime': [
        ("medprime_data_ift", 10),
    ],
    'samples': [
        ("samples_data_ift", 10),
    ]
}


def filter_data(
        results_dir: str,
        result_name: str,
        samples_path: str,
        dataset_key: str,
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    concat_data = list()

    for filename, thresh in tqdm(FILES_TO_FILTER[dataset_key]):
        filepath = f"{samples_path}/{filename}.json"
        print(f"\nParse {filepath} with threshold={thresh}")
        with open(filepath, encoding="utf-8") as r:
            data = json.load(r)
            print(f"Dataset length before removing short samples: {len(data)}")
            data = remove_short_samples(data, min_words_thresh=thresh)
            print(f"Dataset length after removing short samples: {len(data)}")
            concat_data.extend(data)

    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        for record in tqdm(concat_data):
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def remove_short_samples(data: list, min_words_thresh: int = 10):
    new_data = list()
    for record in data:
        words = record["output"].split(" ")
        sentence_len = len([w for w in words if len(w) > 1])
        if sentence_len > min_words_thresh:
            new_data.append(record)
    return new_data


def generate_data() -> None:
    filter_data(
        results_dir="data/data_ift/wikimed_filtered",
        result_name="wikimed_data_all_filtered_ift.jsonl",
        samples_path="data/data_ift/wikimed",
        dataset_key="wikimed",
    )
    filter_data(
        results_dir="data/data_ift/rumedprime_filtered",
        result_name="rumedprime_data_filtered_ift.jsonl",
        samples_path="data/data_ift/rumedprime",
        dataset_key="rumedprime",
    )
    filter_data(
        results_dir="data/data_ift/samples_filtered",
        result_name="samples_data_filtered_ift.jsonl",
        samples_path="data/data_ift/samples",
        dataset_key="samples",
    )


if __name__ == "__main__":
    fire.Fire(generate_data)

import click
import json
import numpy as np
import os
import pandas as pd
import uuid
from functools import partial
from typing import Dict, List


SEED = 17


def get_symptoms_mean(icd_code: str, icd2symptoms: Dict[str, List[str]]) -> float:
    k = 0
    all_symptoms = []
    for code, symptoms in icd2symptoms.items():
        if code[:3] == icd_code:
            all_symptoms += symptoms
            k += 1
    return len(set(all_symptoms)) / k if k else 0


def sample_symptoms(code: str) -> List[str]:
    codes = [c for c in icd2symptoms.keys() if c[:3] == code]
    leader = np.random.choice(codes)
    its_symptoms = icd2symptoms[leader]
    l = len(its_symptoms)
    n_samples = min(l, np.random.randint(2, 6))
    if l > n_samples:
        return leader, "|".join(
            sorted(np.random.choice(its_symptoms, n_samples, replace=False))
        )
    else:
        return leader, "|".join(sorted(its_symptoms))


def check(symptoms: str):
    symptoms = symptoms.split("|")
    return len(symptoms) == len(set(symptoms))


def shuffle_symptoms(s: str) -> stt:
    s = s.split("|")
    np.random.shuffle(s)
    return "|".join(s)


@click.command()
@click.option("--samples-filepath", required=True)
@click.option("--rumedtop3-filepath", required=True)
@click.option("--symptoms-filepath", required=True)
@click.option("--output-folder", required=True)
@click.option("--min-sample-words", default=30, type=int)
def main(
    samples_filepath: str,
    rumedtop3_filepath: str,
    symptoms_filepath: str,
    output_folder: str,
    min_sample_words: int,
):
    with open(symptoms_filepath, "rt") as file:
        icd2symptoms = json.load(file)

    rumed_df = pd.read_json(rumedtop3_filepath, lines=True)
    rumed_codes = set(rumed_df["code"].unique())

    # region Фильтрация реальных сэмплов по числу слов
    samples = pd.read_csv(samples_filepath)

    samples["ICD10_CAT"] = samples["gt"].str[:3]
    samples["n_words"] = samples["raw_visit_text"].str.split().str.len()

    filtered_samples = (
        samples[samples["n_words"] >= min_sample_words]
        .reset_index(drop=True)
        .rename(columns={"raw_visit_text": "anamnesis", "gt": "ICD10_CODE"})
    )
    # endregion

    g = filtered_samples.groupby("ICD10_CAT").size()
    taskgen = pd.DataFrame({"ICD10_CAT": g.index, "size": g.values})
    taskgen = taskgen[taskgen["ICD10_CAT"].isin(rumed_codes)]

    # region Обработка отсутствующих кодов в taskgen из RuMedTop3
    lack_taskgen = set(rumed_codes) - set(taskgen["ICD10_CAT"].values)

    if not len(filtered_samples.loc[filtered_samples["ICD10_CAT"].isin(lack_taskgen)]):
        print(
            "\033[31mОбработка отсутствующих кодов в taskgen из RuMedTop3:",
            lack_taskgen,
        )
        print("Будет использован общий шаблон.\033[0m")
        taskgen = pd.concat(
            [taskgen, pd.DataFrame({"ICD10_CAT": list(lack_taskgen), "size": 1})]
        )
    taskgen["use_template"] = False
    taskgen.loc[taskgen["ICD10_CAT"].isin(lack_taskgen), "use_template"] = True
    # endregion

    calc_symptoms_mean = partial(get_symptoms_mean, icd2symptoms=icd2symptoms)

    taskgen["mean_symptoms"] = taskgen["ICD10_CAT"].map(calc_symptoms_mean)

    taskgen["cutting"] = False
    Z00_mask = taskgen["ICD10_CAT"].str.startswith("Z00")
    taskgen.loc[Z00_mask, "cutting"] = True
    Z00_n_symptoms = taskgen.loc[Z00_mask, "mean_symptoms"].values[0]
    taskgen.loc[Z00_mask, "mean_symptoms"] = 0

    transform = lambda x: np.log1p(np.log1p(np.log1p(x)))

    taskgen["weight"] = transform(taskgen["size"]) * transform(taskgen["mean_symptoms"])
    taskgen["normed_weight"] = taskgen["weight"] / taskgen["weight"].sum()
    taskgen["generate"] = np.int32(np.round(taskgen["normed_weight"] * 2491))

    taskgen.loc[Z00_mask, "mean_symptoms"] = Z00_n_symptoms
    taskgen.loc[Z00_mask, "generate"] = 10

    taskgen.sort_values(by="size", ascending=False, inplace=True)
    taskgen.to_csv(os.path.join(output_folder, "taskgen.csv"), index=False)

    # region
    np.random.seed(SEED)

    rumed_samples = filtered_samples[filtered_samples["ICD10_CAT"].isin(rumed_codes)]
    template_taskgen_df = pd.DataFrame({"ICD10_CAT": list(lack_taskgen)})
    template_taskgen_df["anamnesis"] = "Пациент обратился с жалобами."
    template_taskgen_df["agent"] = -1
    template_taskgen_df["visit"] = -1
    template_taskgen_df["gender"] = -1
    template_taskgen_df["age"] = -1
    template_taskgen_df["ICD10_CODE"] = template_taskgen_df["ICD10_CAT"]
    rumed_samples = pd.concat([rumed_samples, template_taskgen_df])

    dfs = []
    for e in taskgen.itertuples():
        dfs.append(
            rumed_samples.loc[rumed_samples["ICD10_CAT"] == e.ICD10_CAT].sample(
                e.generate, replace=True, random_state=SEED
            )
        )
    # endregion

    sampled_taskgen = pd.concat(dfs)
    sampled_taskgen["anamnesis"] = (
        sampled_taskgen["anamnesis"]
        .str.replace("\n", " ", regex=False)
        .str.replace("\r", "", regex=False)
        .str.replace(r"\s{2,}", " ", regex=True)
    )

    sampled_taskgen["symptoms_donor"], sampled_taskgen["sampled_symptoms"] = zip(
        *sampled_taskgen["ICD10_CAT"].map(sample_symptoms)
    )
    sampled_taskgen.drop(
        columns=["index", "hist", "word_count", "text_length"], inplace=True
    )
    sampled_taskgen["duplicated"] = sampled_taskgen.duplicated(keep=False)
    sampled_taskgen["sample_id"] = [uuid.uuid4() for k in range(len(sampled_taskgen))]

    columns = [sampled_taskgen.columns[-1]] + sampled_taskgen.columns[:-1].tolist()
    sampled_taskgen = sampled_taskgen[columns]

    assert all(sampled_taskgen["sampled_symptoms"].map(check))

    np.random.seed(SEED)
    sampled_taskgen["shuffled_symptoms"] = sampled_taskgen["sampled_symptoms"].map(
        shuffle_symptoms
    )

    sampled_taskgen.to_csv(
        os.path.join(output_folder, "sampled_taskgen.csv"), index=False
    )


if __name__ == "__main__":
    main()

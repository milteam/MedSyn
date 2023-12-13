"""Convert WikiMed data in 1-3 hops questions to IFT format."""

from typing import Dict, List

import os
import json
import click
import random
import numpy as np
import pandas as pd
from os.path import join


PRE = [
    "",
    "Ты являешься врачом.",
    "Ты эксперт в области медицины.",
]


def get_3_hop_squestion(
    symptoms: Dict, disiases: pd.DataFrame, max_drugs: int = 3
) -> List:
    instructions = [
        "Перечисли медикаменты, которые могут принимать при заболевании если оно ошибочно принято за другое заболевание со схожими симптомами.",
        "Укажи лекарства, которые могут быть назначены для лечения болезни, когда она неверно диагностирована как другое заболевание с похожими признаками.",
        "Назови препараты, применяемые при заболевании, которое может быть ошибочно идентифицировано как другая болезнь с аналогичными симптомами.",
        "Перечисли лекарственные средства, используемые в случае, когда болезнь неправильно расценена как другое заболевание с сходными проявлениями.",
        "Опиши медикаменты, применимые при диагностировании болезни, которая может быть ошибочно воспринята за другую с похожими симптомами.",
        "Укажи фармацевтические препараты, которые могут использоваться при лечении заболевания, если оно неверно диагностировано как болезнь с схожей симптоматикой.",
    ]

    common_dis = {}
    for d1, s1 in symptoms.items():
        for d2, s2 in symptoms.items():
            common = list(set(s1).intersection(set(s2)))
            if common and d1 != d2:
                if (d1, d2) not in common_dis:
                    common_dis[(d1, d2)] = common

    result = []
    for dis, symp in common_dis.items():
        d1, d2 = dis

        d1_name = disiases[disiases["МКБ-10"] == d1]["Рубрика"]
        d2_name = disiases[disiases["МКБ-10"] == d2]["Рубрика"]

        instruction = random.choice(instructions)
        pre = random.choice(PRE)
        instruction = pre + " " + instruction if pre else instruction

        if d1_name.shape[0] > 0 and d2_name.shape[0] > 0:
            drugs_by_ids = disiases[disiases["МКБ-10"] == d2]["Действующие вещества"]

            if len(drugs_by_ids) > 0:
                d1_name = d1_name.values[0]
                d2_name = d2_name.values[0]
                drugs_by_ids = drugs_by_ids.values[0]

                if isinstance(drugs_by_ids, str):
                    drugs_by_ids = drugs_by_ids.split("\n")

                    input = f"Заболевание {d1_name}."
                    if len(drugs_by_ids) > max_drugs:
                        for _ in range(max_drugs):
                            drugs = np.random.choice(
                                drugs_by_ids, size=max_drugs, replace=False
                            )
                            drugs = ", ".join(set(drugs))
                            output = f"Заболевание {d1_name} имеет общие симптомы с заболеванием {d2_name}, поэтому могут приниматься медикаменты назначаемые при {d2_name}, например: {drugs}."
                            sample = {
                                "instruction": instruction,
                                "input": input,
                                "output": output,
                            }
                            result.append(sample)
                    else:
                        drugs = ", ".join(set(drugs_by_ids))
                        output = f"Заболевание {d1_name} имеет общие симптомы с заболеванием {d2_name}, поэтому могут приниматься медикаменты назначаемые при {d2_name}, например: {drugs}."
                        sample = {
                            "instruction": instruction,
                            "input": input,
                            "output": output,
                        }
                        result.append(sample)
    return result


def get_2_hop_squestion(
    symptoms: Dict, disiases: pd.DataFrame, max_sympt: int = 5
) -> List:
    instructions = [
        "Напиши медикаменты которые могут принимать при данных симптомах.",
        "Перечисли лекарства, рекомендуемые для лечения этих симптомов.",
        "Укажи препараты, подходящие для устранения указанных симптомов.",
        "Опиши медикаментозные средства, применимые при наличии этих симптомов.",
        "Назови лекарственные препараты, которые используются для этих симптомов.",
        "Составь список медикаментов, подходящих для данных симптомов.",
    ]
    outputs = [
        "Данные симптомы могут свидетельствовать о наличии заболевания {}, поэтому могут принимать препарат {}.",
        "Эти симптомы могут указывать на заболевание {}, в связи с чем рекомендуется препарат {}.",
        "При наличии таких симптомов возможно наличие {}, для лечения которого подходит {}.",
        "Эти симптомы часто ассоциируются с {}, и в таких случаях обычно назначают {}.",
        "Если наблюдаются эти симптомы, это может быть признаком {}, и в этом случае может помочь {}.",
        "Учитывая эти симптомы, вероятно, у вас {}, для чего обычно применяют {}.",
    ]

    result = []

    for dis, symp in symptoms.items():
        dis_name = disiases[disiases["МКБ-10"] == dis]["Рубрика"]
        drugs_by_ids = disiases[disiases["МКБ-10"] == dis]["Действующие вещества"]

        instruction = random.choice(instructions)
        pre = random.choice(PRE)
        instruction = pre + " " + instruction if pre else instruction

        if len(dis_name) > 0 and len(drugs_by_ids) > 0:
            dis_name = dis_name.values[0]
            drugs_by_ids = drugs_by_ids.values[0]

            if dis_name is not None and isinstance(drugs_by_ids, str):
                drugs_by_ids = drugs_by_ids.split("\n")

                if len(symp) > max_sympt:
                    for _ in range(2 * max_sympt):
                        symp_ = list(
                            np.random.choice(symp, size=len(symp) // 2, replace=False)
                        )[:max_sympt]
                        drug = np.random.choice(drugs_by_ids)
                        output = random.choice(outputs)
                        output = output.format(dis_name, drug)
                        symp_ = ", ".join(set(symp_))
                        sample = {
                            "instruction": instruction,
                            "input": symp_,
                            "output": output,
                        }
                        result.append(sample)
                else:
                    drug = np.random.choice(drugs_by_ids)
                    output = random.choice(outputs)
                    output = output.format(dis_name, drug)
                    sample = {
                        "instruction": instruction,
                        "input": symp,
                        "output": output,
                    }
                    result.append(sample)

    return result


def get_1_hop_squestion_dis(symptoms: Dict, disiases: pd.DataFrame) -> List:
    instructions = [
        "Напиши общие симптомы для двух заболеваний.",
        "Перечисли симптомы, которые являются общими для обоих заболеваний.",
        "Опиши клинические признаки, характерные одновременно для двух разных болезней.",
        "Укажи симптоматику, которая обычно встречается в обеих из этих болезней.",
        "Сформулируй список проявлений, общих для этих двух заболеваний.",
        "Запиши характерные признаки, которые могут наблюдаться при обеих болезнях.",
    ]

    ans_format = [
        "Заболевания {} и {} имеют следующие общие симптомы: {}.",
        "У заболеваний {} и {} наблюдаются такие же симптомы: {}.",
        "Для болезней {} и {} характерны общие клинические признаки: {}.",
        "Среди симптомов, общих для {} и {}, можно выделить: {}.",
        "Болезни {} и {} проявляются через следующие схожие симптомы: {}.",
        "Общие проявления для заболеваний {} и {} включают: {}.",
    ]

    instruction = random.choice(instructions)
    pre = random.choice(PRE)
    instruction = pre + " " + instruction if pre else instruction

    common_dis = {}

    for d1, s1 in symptoms.items():
        for d2, s2 in symptoms.items():
            common = list(set(s1).intersection(set(s2)))
            if common and d1 != d2:
                if (d1, d2) not in common_dis and (d2, d1) not in common_dis:
                    common_dis[(d1, d2)] = common

    result = []
    for dis, symp in common_dis.items():
        d1, d2 = dis

        d1_name = disiases[disiases["МКБ-10"] == d1]["Рубрика"]
        d2_name = disiases[disiases["МКБ-10"] == d2]["Рубрика"]

        if d1_name.shape[0] > 0 and d2_name.shape[0] > 0:
            d1_name = d1_name.values[0]
            d2_name = d2_name.values[0]
            names = f"{d1_name} и {d2_name}"
            output = ", ".join(set(symp))
            ans = random.choice(ans_format).format(d1_name, d2_name, output)
            sample = {"instruction": instruction, "input": names, "output": ans}
            result.append(sample)

    return result


def get_1_hop_squestion_drugs(drugs: pd.DataFrame) -> List:
    instructions = [
        "Напиши к какой фармакологической группе относятся медикаменты.",
        "Укажи фармакологическую категорию, к которой принадлежат эти медикаменты.",
        "Опиши к какой группе лекарственных средств относится данный препарат.",
        "Классифицируй эти медикаменты по их фармакологической группе.",
        "Определи к какому классу фармакологии принадлежат эти лекарства.",
        "Уточни группу лекарственных средств, к которой относятся эти медикаменты.",
    ]

    ans_format = [
        "Данные препараты относятся к фармокологической группе {}",
        "Эти медикаменты классифицированы как принадлежащие к фармакологической группе {}.",
        "Препараты, о которых идет речь, входят в группу фармакологии {}.",
        "Указанные лекарства являются частью фармакологической категории {}.",
        "Эти лекарственные средства относятся к классу {} в фармакологии.",
        "Препараты, описанные здесь, принадлежат к фармакологической группе {}.",
    ]

    instruction = random.choice(instructions)
    pre = random.choice(PRE)
    instruction = pre + " " + instruction if pre else instruction

    wikimed_data = drugs.dropna(
        subset=["Фармакологическая группа"], axis=0, inplace=False
    )

    result = []
    for idx in wikimed_data.index:
        data = wikimed_data.loc[idx]
        group = data["Фармакологическая группа"]
        meds_list = wikimed_data[wikimed_data["Фармакологическая группа"] == group]
        names = list(meds_list["Название"].values)
        if len(names) > 1:
            names = ", ".join(set(names))
            output = random.choice(ans_format).format(group)
            sample = {"instruction": instruction, "input": names, "output": output}
            result.append(sample)

    return result


@click.command()
@click.option("--wikimed-path", default="data/data_raw/wikimed/wikimed_diseases.csv")
@click.option("--wikimed-drugs-path", default="data/data_raw/wikimed/wikimed_meds.csv")
@click.option(
    "--symptoms-path",
    default="data/data_raw/rumedtop3_wikimed_symptoms_equipped_with_rumedprime.json",
)
@click.option(
    "--results-dir",
    default="data/data_ift/wikimed_graph/",
)
def generate_data(
    wikimed_path: str, wikimed_drugs_path: str, symptoms_path: str, results_dir: str
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    with open(symptoms_path, "rb") as f:
        symptoms = json.load(f)

    disiases = pd.read_csv(wikimed_path)
    drugs = pd.read_csv(wikimed_drugs_path)

    result = []

    samples = get_3_hop_squestion(symptoms, disiases)
    result.extend(samples)

    samples = get_2_hop_squestion(symptoms, disiases)
    result.extend(samples)

    samples = get_1_hop_squestion_dis(symptoms, disiases)
    result.extend(samples)

    samples = get_1_hop_squestion_drugs(drugs)
    result.extend(samples)

    print(f"Prepared {len(result)} samples.")

    with open(join(results_dir, "wikimed_graph.json"), "w", encoding="utf8") as f:
        json.dump(result, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    generate_data()

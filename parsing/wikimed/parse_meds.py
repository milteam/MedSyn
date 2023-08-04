import json
import argparse
from bs4 import BeautifulSoup, NavigableString, Tag
import requests
from tqdm import tqdm
import pandas as pd
from typing import Dict


def parse(soup: BeautifulSoup) -> Dict:
    out = {}
    head = soup.find(["h1"]).text
    for header in soup.find_all(["h3"]):
        nextNode = header

        FIELDS = [
            "Латинское название[править]",
            "Фармакологическая группа[править]",
            "Характеристика вещества[править]",
            "Фармакология[править]",
            "Применение[править]",
            f"{head}: Противопоказания[править]",
            "Применение при беременности и кормлении грудью[править]",
            f"{head}: Побочные действия[править]",
            "Взаимодействие[править]",
            f"{head}: Способ применения и дозы[править]",
            "Меры предосторожности[править]",
            "Торговые наименования[править]",
            "МКБ-10[править]",
        ]

        if nextNode.text not in FIELDS:
            continue

        key = nextNode.text
        content = []

        while True:
            nextNode = nextNode.nextSibling
            if nextNode is None:
                break
            if isinstance(nextNode, NavigableString):
                continue
            if isinstance(nextNode, Tag):
                if nextNode.name == "h2":
                    break
                if nextNode.name == "h3":
                    break
                if nextNode.name == "div":
                    continue
                elif len(nextNode.get_text(strip=True).strip()) != 0:
                    content.append(nextNode.get_text(strip=False).strip())
        if content:
            out[key] = ";".join(content)

    result = []
    for field in FIELDS:
        if field not in out:
            result.append(None)
        else:
            result.append(out[field] if out[field] != field else None)

    return result


def main():
    parser = argparse.ArgumentParser(description="Parse medicaments from WikiMed.")
    parser.add_argument("--all_data", default="data/wikimed/meds.json")
    parser.add_argument("--path_to_save", default="data/wikimed/wikimed_meds.csv")

    args = parser.parse_args()

    with open(args.all_data, "r") as f:
        diseases = json.load(f)

    parsed_data = pd.DataFrame(
        columns=[
            "Название",
            "Латинское название",
            "Фармакологическая группа",
            "Характеристика вещества",
            "Фармакология",
            "Применение",
            "Противопоказания",
            "Применение при беременности и кормлении грудью",
            "Побочные действия",
            "Взаимодействие",
            "Способ применения и дозы",
            "Меры предосторожности",
            "Торговые наименования",
            "МКБ-10",
            "url",
        ]
    )
    idx = 0
    for data in tqdm(diseases):
        name, url = data["name"], data["link"]

        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        data = parse(soup)

        parsed_data.loc[idx] = [name] + data + [url]
        idx += 1

    parsed_data.to_csv(args.path_to_save, index=False)


if __name__ == "__main__":
    main()

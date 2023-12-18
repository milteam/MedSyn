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
            "Определение и общие сведения[править]",
            "Этиология и патогенез[править]",
            "Клинические проявления[править]",
            f"{head}: Диагностика[править]",
            "Дифференциальный диагноз[править]",
            f"{head}: Лечение[править]",
            "Профилактика[править]",
            "Действующие вещества[править]",
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
    parser = argparse.ArgumentParser(description="Parse diseases from WikiMed.")
    parser.add_argument("--diseases", default="data/wikimed/all_data.json")
    parser.add_argument("--path_to_save", default="data/wikimed/wikimed_diseases.csv")

    args = parser.parse_args()

    with open(args.diseases, "r") as f:
        diseases = json.load(f)

    parsed_data = pd.DataFrame(
        columns=[
            "МКБ-10",
            "Рубрика",
            "Определение и общие сведения",
            "Этиология и патогенез",
            "Клинические проявления",
            "Диагностика",
            "Дифференциальный диагноз",
            "Лечение",
            "Профилактика",
            "Действующие вещества",
            "url",
        ]
    )
    idx = 0
    for data in tqdm(diseases):
        name, secondary_name, url = data["name"], data["secondary_name"], data["link"]

        if secondary_name and "Рубрика МКБ-10:" in secondary_name:
            icd = secondary_name.split(":")[-1].strip()
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, "html.parser")
            data = parse(soup)

            parsed_data.loc[idx] = [icd, name] + data + [url]
            idx += 1

    parsed_data.to_csv(args.path_to_save, index=False)


if __name__ == "__main__":
    main()

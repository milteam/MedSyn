import re
import json
import argparse
from bs4 import BeautifulSoup, NavigableString, Tag
import requests
from tqdm import tqdm
import pandas as pd
from typing import Dict


def parse(soup: BeautifulSoup) -> Dict:
    out = {}
    current_header_tag = None

    for header in soup.find_all(["h2", "h3"]):
        current_header_tag = header.name
        nextNode = header

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
                if current_header_tag == "h2" and nextNode.name == "h3":
                    break
                elif len(nextNode.get_text(strip=True).strip()) != 0:
                    content.append(nextNode.get_text(strip=False).strip())
        if content:
            content = ";".join(content)
            content = re.sub(r"\s+", " ", content)
            key = re.sub(r"\s+", " ", key).strip()
            out[key] = content

    res = {}

    for k1, v1 in out.items():
        for k2, v2 in out.items():
            if k2 != k1:
                if k2 + " " + v2 in v1:
                    v1 = v1.replace(k2 + " " + v2, "")
        res[k1] = v1

    return res


def main():
    parser = argparse.ArgumentParser(description="Parse symptoms from MSD.")
    parser.add_argument("--symptoms", default="data/msd/symptoms.json")
    parser.add_argument("--path_to_save", default="data/msd/msd_symptoms.json")

    args = parser.parse_args()

    results = {}

    with open(args.symptoms, "r") as f:
        symptoms = json.load(f)

    for data in tqdm(symptoms):
        name, url = data["name"], data["link"]

        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        data = parse(soup)

        name = name.strip()
        results[name] = data

    with open(args.path_to_save, "w", encoding="utf8") as f:
        json.dump(results, f, indent=3, ensure_ascii=False)


if __name__ == "__main__":
    main()

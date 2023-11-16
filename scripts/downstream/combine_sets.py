import json

import click
import pandas as pd
from sklearn.model_selection import train_test_split


def load(path):
    df = []
    with open(path) as f:
        for row in f.readlines():
            df.append(json.loads(row))

    return pd.DataFrame(df)

@click.command()
@click.option("--train", "-t", help="First dataset", required=True)
@click.option("--pt", help="Proportion of first dataset", type=float, required=True)
@click.option("--upsample", "-u", help="Second dataset", required=False)
@click.option("--pu", help="Proportion of second dataset", type=float, required=False, default=0)
@click.option("--out", help="Output file", type=str, required=True)
def main(train, pt, upsample, pu, out):
    combine(train, pt, upsample, pu, out)

def combine(train, pt, upsample, pu, out):
    data = load(train)
    if pt < 1:
        _, data = train_test_split(data, test_size=pt, stratify=data[["code"]])
    if pu > 0:
        second = load(upsample)
        if pu < 1:
            _, second = train_test_split(second, test_size=pu, stratify=second[["code"]])
        data = pd.concat([data, second], ignore_index=True)

    data.to_json(out, orient='records', lines=True, force_ascii=False)


# Train Dataset	Downsampling	Upsampling dataset	Upsampling
# train_v1_symp_anam	0,5	-	0
# train_v1_symp_anam	0,75	-	0
# train_v1_symp_anam	1	-	0
# train_v1_symp_anam	0,5	ChatGPT4	0,5
# train_v1_symp_anam	0,5	ChatGPT4	1
# train_v1_symp_anam	1	ChatGPT4	0,25
# train_v1_symp_anam	1	ChatGPT4	0,5
# train_v1_symp_anam	1	ChatGPT4	1
# train_v1_symp_anam	0	ChatGPT4	1
def combine_all_gpt4():
    train = "scripts/downstream/data/RuMedTop3/train_v1_symp_anam.jsonl"
    upsample = "scripts/downstream/data/RuMedTop3/gpt4_train.jsonl"

    proportions = [[0.5, 0], [0.75, 0], [0.5, 0.5], [0.5, 1], [1, 0.25], [1, 0.5], [1, 1]]

    for pt, pu in proportions:
        combine(train, pt, upsample, pu, f"scripts/downstream/data/RuMedTop3/gpt4/sa{int(pt*100)}_gpt{int(pu*100)}.jsonl")

if __name__ == '__main__':
    combine_all_gpt4()
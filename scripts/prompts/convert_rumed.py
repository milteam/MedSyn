import json
import re
import uuid

from nltk import word_tokenize

MAX_LEN = 256
MIN_LEN = 3
text1_max = int(MAX_LEN * .75)  # leave 75% of token lens to premise text
text2_max = MAX_LEN - text1_max


def check_sentence(sentence):
    if len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', sentence)) == 0:
        return False

    if ":" in sentence:
        #print("Colon in sentence")
        return False
    sl = len(word_tokenize(sentence))
    if sl > MAX_LEN * .6 or sl < MIN_LEN:
        print("Long or short sentence")
        return False

    return True


if __name__ == '__main__':
    files = ["nli_results.json", "nli_results_3.json", "nli_results_0.json", "nli_results_1.json", "nli_results_2.json"]
    res = []
    filtered = 0
    for file in files:
        with open(file) as f:
            data = json.load(f)

        for item in data:
            # if len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', item["original"])) == 0 or len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', item["entailment"])) == 0 or \
            #         len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', item["neutral"])) == 0 or len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', item["contradiction"])) == 0:
            #     print("Filtered empty ", item)
            #     continue
            if all([check_sentence(s) for s in item.values()]):
                res.append({"ru_sentence1": item["original"], "ru_sentence2": item["entailment"], "gold_label": "entailment", "pairID": str(uuid.uuid4())})
                res.append({"ru_sentence1": item["original"], "ru_sentence2": item["contradiction"], "gold_label": "contradiction", "pairID": str(uuid.uuid4())})
                res.append({"ru_sentence1": item["original"], "ru_sentence2": item["neutral"], "gold_label": "neutral", "pairID": str(uuid.uuid4())})
            else:
                filtered += 1

    print("Filtered ", filtered)

    with open("data/train_v1.jsonl") as f:
        lines = f.readlines()
    with open("train_v1.jsonl", "w") as f:
        f.writelines(lines)
        f.writelines([f"{json.dumps(item, ensure_ascii=False)}\n" for item in res])


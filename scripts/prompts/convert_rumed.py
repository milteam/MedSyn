import json
import random
import re
import uuid
import re
from nltk import word_tokenize

MAX_LEN = 256
MIN_LEN = 3
text1_max = int(MAX_LEN * .75)  # leave 75% of token lens to premise text
text2_max = MAX_LEN - text1_max

# Бели обычно 1 раз в неделю, в основном пиво
#
# Даты - 2 из 200
# {"ru_sentence1": "Пациент курит около 20 сигарет в день с 18 лет, употребляет алкоголь – раз в неделю, но в больших количествах.", "ru_sentence2": "С учетом такого образа жизни, его шансы на развитие кардиоваскулярных и онкологических заболеваний значительно возрастают.", "gold_label": "entailment", "pairID": "354785d0-b885-4081-a76b-dc31edcb26bd"}
# Работает

stop_words = ["курит", "курен", "работает", "алкого", "пациент:", "образ жизни"]


def failed_sentence(sentence):
    sentence = sentence.lower()

    if len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', sentence)) == 0:
        return True

    if re.search(r'\d{4}', sentence) is not None:
        print("Found date  ", sentence)
        return True

    for stop_word in stop_words:
        if stop_word in sentence:
            print("Found stop word ###START\n  ", sentence, "###END")
            return True

    sl = len(word_tokenize(sentence))
    if sl > MAX_LEN * .6 or sl < MIN_LEN:
        #print("Long or short sentence")not
        return True

    # if ":" in sentence:
    #     #print("Colon in sentence")
    #     return True

    return False


if __name__ == '__main__':
    files = ["nli_results.json", "nli_results_3.json", "nli_results_0.json", "nli_results_1.json", "nli_results_2.json"]
    mined = []
    filtered = 0
    for file in files:
        with open(file) as f:
            data = json.load(f)

        for item in data:
            # if len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', item["original"])) == 0 or len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', item["entailment"])) == 0 or \
            #         len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', item["neutral"])) == 0 or len( re.sub('[^0-9a-zA-Zа-яА-Я]+', '', item["contradiction"])) == 0:
            #     print("Filtered empty ", item)
            #     continue
            if not any([failed_sentence(s) for s in item.values()]):#not failed_sentence(item["original"]):#
                mined.append(item)
            else:
                filtered += 1

    print("Filtered ", filtered)
    print("Remaining ", len(mined))

    #mined = random.sample(mined, 1000)
    res =  []
    for item in mined:
        res.append({"ru_sentence1": item["original"], "ru_sentence2": item["entailment"], "gold_label": "entailment",
                    "pairID": str(uuid.uuid4())})
        res.append(
            {"ru_sentence1": item["original"], "ru_sentence2": item["contradiction"], "gold_label": "contradiction",
             "pairID": str(uuid.uuid4())})
        res.append({"ru_sentence1": item["original"], "ru_sentence2": item["neutral"], "gold_label": "neutral",
                    "pairID": str(uuid.uuid4())})

    with open("data/train_v1.jsonl") as f:
        lines = []#f.readlines()
    with open("train_v1.jsonl", "w") as f:
        f.writelines(lines)
        f.writelines([f"{json.dumps(item, ensure_ascii=False)}\n" for item in res])


import argparse
import itertools
import json
import os
import re
from functools import partial
from glob import glob
from pathlib import Path
from typing import Iterable, Literal, TypedDict

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from razdel import sentenize
from sentence_transformers import SentenceTransformer, util


GenderType = Literal["male", "female"]


ICD_CODE = str


class SyntheticRecord(TypedDict, total=True):
    UID: str
    gender: GenderType
    desease_code: ICD_CODE
    response: str


def load_file(path: str):
    with open(path, "rt") as file:
        d = json.load(file)
    filename = Path(path).name
    for e in d:
        e["filename"] = filename
    return d


def isGenderOK(record: SyntheticRecord, gen: Iterable[ArrayLike]):
    n_sentences = record["n_sentences"]
    scores = np.vstack(list(itertools.islice(gen, n_sentences)))
    diff = scores[:, 0] - scores[:, 1]
    something_wrong = (record["gender"] == "female" and any(diff >= 0.035)) or (record["gender"] == "male" and any(diff <= -0.07))
    return not something_wrong


_speech_tokens = ["?", "!", "добрый день", "здравствуй", "спасибо", "пожалуйста", " я ",
    " расскажи", " давайте", " вас ", " вы ", " мне ", " меня ", " мой ", " мою ",
    "начнем", "стесняйтесь", "стесняться"]


def isSpeech(s: str) -> bool:
    return any(token in s.lower() for token in _speech_tokens) or " ваш" in s or " Ваш" in s


field_expr = re.compile(r'\[.+\]')
angle_expr = re.compile(r'<.+>')


def isForm(s: str) -> bool:
    # Except for: (ФИО), (возраст), ...
    s = s.lower().replace("<<", "\"")
    return ('__' in s) or ('██' in s) or ("подпись" in s) or ("печать" in s) or \
        (field_expr.search(s) is not None) or (angle_expr.search(s) is not None)


def main():
    #region Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, required=True, help="Директория, содержащая JSON-файлы с параметрами и результатами генерации.")
    parser.add_argument("--output-path", type=str, required=True, help="Пусть к CSV-таблице с результатами фильтрации.")
    args = vars(parser.parse_args())
    #endregion

    INPUT_FOLDER = args["input_folder"]
    OUTPUT_PATH = args["output_path"]
    # Кол-во извлекаемых предложений для анализа гендера.
    N_SENTENCES = 3

    data = [load_file(filename) for filename in glob(os.path.join(INPUT_FOLDER, "*.json"))]
    data = list(itertools.chain(*data))

    model = SentenceTransformer("ai-forever/sbert_large_mt_nlu_ru")

    #region Sentence Preprocessing
    sentences = [[s.text for s in itertools.islice(sentenize(record["response"]), N_SENTENCES)] for record in data]

    for r, s in zip(data, sentences):
        r["n_sentences"] = len(s)

    sentences = list(itertools.chain(*sentences))
    sentence_emb = model.encode(sentences, convert_to_tensor=True)

    gender_prompts = ["Мужчина обратился с жалобой.", "Женщина обратилась с жалобой."]
    gender_emb = model.encode(gender_prompts, convert_to_tensor=True, show_progress_bar=False)
    #endregion

    scores = util.cos_sim(sentence_emb, gender_emb)
    scores = scores.cpu().numpy()

    check_gender = partial(isGenderOK, gen=(row for row in scores))

    record_info = [{
        "UID": r["UID"],
        "STATUS": 1,
        "WRONG_GENDER": int(check_gender(r)),
        "SPEECH": int(isSpeech(r["response"])),
        "FORM": int(isForm(r["response"])),
        "FILENAME": r["filename"]
    } for r in data]

    df = pd.DataFrame(record_info)
    df["STATUS"] = np.uint8(df[["WRONG_GENDER", "SPEECH", "FORM"]].sum(axis=1) == 0)
    df.to_csv(OUTPUT_PATH, index=None)


if __name__ == "__main__":
    main()
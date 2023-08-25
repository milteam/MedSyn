import argparse
import itertools
import json
import os
import re
from functools import partial
from glob import glob
from pathlib import Path
from typing import Iterable, Literal, TypedDict

import cupy as cp
import numpy as np
import pandas as pd
from cuml.manifold import UMAP
from cuml.svm import SVC as SVC
from numpy.typing import ArrayLike
from razdel import sentenize
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

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

#region FICTION FILTERING
_fiction_tokens = [" вымышлен", " выдуман", " фантаз", " виртуальн", " реальн", " гипотетичес", " фиктив", " фикци", \
    " демонстрац", " художеств", " квалифицир", " симул", " надуман", \
    " educational", " не может заменить", " условн", " иллюстр", "openai", "gpt", " ии", " анамнез представлен справочно"]


edu_expr = re.compile(r"(образовательн|учеб|информацион|ознакомительн)\w+\s(характер|цел)")
nomed_expr = re.compile(r"не является (медицинской )?(консультацией|советом)")
llm_expr = re.compile(r"(языков\w+ модел)|(искусствен\w+ интеллект)")


def isFictionSentence(sentence: str) -> bool:
    return any(token in " " + sentence.lower() for token in _fiction_tokens) or \
        (llm_expr.search(sentence) is not None) or \
        (edu_expr.search(sentence) is not None) or \
        (nomed_expr.search(sentence) is not None)


def isFictionSentences(sentences: Iterable[str], normalized_emb):
    labels = np.array([isFictionSentence(s) for s in sentences], dtype=np.int32)

    N_COMPONENTS = 4
    REG_PARAM = 2
    SEED = 42

    for k in range(4):
        pred = np.zeros_like(labels, dtype=np.int32)
        for k in tqdm(range(15)):
            reductor = UMAP(n_components=N_COMPONENTS, random_state=SEED, metric="cosine")
            X = cp.asnumpy(reductor.fit_transform(normalized_emb))

            clf = SVC(random_state=SEED, C=REG_PARAM)
            clf.fit(X, labels)
            pred += clf.predict(X)

        pred = np.int32(pred > 0)

        mask = labels - pred >= 0
        if all(mask):
            break
        labels[~mask] = 1

    return labels


def isFiction(record: SyntheticRecord, gen: Iterable[np.uint8]):
    n_tail_sentences = record["n_tail_sentences"]
    return any(list(itertools.islice(gen, n_tail_sentences)))
#endregion


def isGenderOK(record: SyntheticRecord, gen: Iterable[ArrayLike]):
    n_sentences = record["n_sentences"]
    scores = np.vstack(list(itertools.islice(gen, n_sentences)))
    diff = scores[:, 0] - scores[:, 1]
    something_wrong = (record["gender"] == "female" and any(diff >= 0.035)) or (record["gender"] == "male" and any(diff <= -0.07))
    return not something_wrong


_speech_tokens = ["?", "!", "добрый день", "здравствуй", "спасибо", "пожалуйста", " я ",
    " расскажи", " давайте", " вас ", " вы ", " мне ", " меня ", " мой ", " мою ", " моя ", " моё ", " мое ",
    " нам ", " нас ", "предлагаю", " прошу", " хорошо,", "с удовольствием", "интересно,", " проведу",
    "начнем", "стесняйтесь", "стесняться", "извините", "уважаемый", "благодарю", "позвольте"]


def countSpeechTokens(s: str) -> bool:
    return sum(s.lower().count(token) for token in _speech_tokens) + s.count(" ваш") + s.count(" Ваш")


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
    parser.add_argument("--output-path", type=str, required=True, help="Путь к CSV-таблице с результатами фильтрации.")
    args = vars(parser.parse_args())
    #endregion

    INPUT_FOLDER = args["input_folder"]
    OUTPUT_PATH = args["output_path"]
    # Кол-во извлекаемых предложений для анализа гендера.
    N_SENTENCES = 3
    N_TAIL_SENTENCES = 5

    data = [load_file(filename) for filename in glob(os.path.join(INPUT_FOLDER, "*.json"))]
    data = list(itertools.chain(*data))

    model = SentenceTransformer("ai-forever/sbert_large_mt_nlu_ru")

    #region Sentence Preprocessing
    for r in data:
        r["sentences"] = [s.text for s in sentenize(r["response"])]
        r["n_tail_sentences"] = len(r["sentences"][-N_TAIL_SENTENCES:])

    tail_sentences = [r["sentences"][-N_TAIL_SENTENCES:] for r in data]
    tail_sentences = list(itertools.chain(*tail_sentences))

    tail_sentence_emb = model.encode(tail_sentences, convert_to_tensor=True)
    tail_sentence_norms = tail_sentence_emb.norm(p=2, dim=1, keepdim=True)
    tail_sentence_emb = tail_sentence_emb.div(tail_sentence_norms)

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

    fiction_labels = isFictionSentences(tail_sentences, tail_sentence_emb)
    isFictionRecord = partial(isFiction, gen=(label for label in fiction_labels))

    record_info = [{
        "UID": r["UID"],
        "STATUS": 1,
        "WRONG_GENDER": int(not check_gender(r)),
        "FICTION": int(isFictionRecord(r)),
        "SPEECH@1": countSpeechTokens(r["response"]),
        "FORM": int(isForm(r["response"])),
        "FILENAME": r["filename"]
    } for r in data]

    df = pd.DataFrame(record_info)
    df["SPEECH@4"] = np.uint8(df["SPEECH@1"] >= 4)
    df["SPEECH@1"] = np.uint8(df["SPEECH@1"] >= 1)
    df["STATUS"] = np.uint8(df[["WRONG_GENDER", "SPEECH@4", "FORM", "FICTION"]].sum(axis=1) == 0)

    ordered_columns = ["UID", "STATUS", "WRONG_GENDER", "FICTION", \
        "SPEECH@1", "SPEECH@4", "FORM", "FILENAME"]
    df[ordered_columns].to_csv(OUTPUT_PATH, index=None)


if __name__ == "__main__":
    main()
import json
import os

import fire
import pandas as pd
import xmltodict
from tqdm import tqdm


def generate_data(
        results_dir: str = "data/data_ift/rumedtest_sogma",
        result_name: str = "rumedtest_data_ift.jsonl",
        samples_path: str = "data/data_raw/sogma_test_complete.xml",
) -> None:
    os.makedirs(results_dir, exist_ok=True)

    with open(samples_path, 'r', encoding="utf-8") as f:
        xx = f.read()

    dd = xmltodict.parse(xx)

    qq = []
    for q in tqdm(dd['geetest']['test']['questions']['question']):
        aa = dict(text=q['text'], theme_id=int(q["@idTheme"]))
        for a in q['answers']['answer']:
            aa[a['@num']] = a['#text']
        correct_idx = [a['@num'] for a in q['answers']['answer'] if a['@isCorrect'] == '1']
        try:
            correct_idx = int(correct_idx[0])
        except IndexError:
            correct_idx = 0

        aa['correct'] = correct_idx

        qq.append(aa)

    dfa0 = pd.DataFrame(qq)

    def remove_multiline_questions(s: str) -> bool:
        tt = '123'
        return all(t in s for t in tt)

    dfa0['lines'] = dfa0.text.apply(lambda x: remove_multiline_questions(x))

    dfa = dfa0[dfa0.lines == 0] \
        .drop(columns=['6', 'lines']) \
        .rename(columns={'text': 'question'}) \
        .sample(frac=1, random_state=42, ignore_index=True) \
        .reset_index(drop=True)  # keep only 1-line questions, drop option 6(unused)

    instr_template = "Ты являешься профессиональным врачом. Тебе нужно пройти тест и ответить, какое утверждение является верным. В ответе обязательно напиши один правильный вариант."
    inpt = """
1. {0} {1}. 
2. {0} {2}. 
3. {0} {3}. 
4. {0} {4}. 
5. {0} {5}."""
    output_template = "Ответ: {0}. {1} {2}."
    with open(os.path.join(results_dir, result_name), "w", encoding="utf-8") as w:
        for i in tqdm(range(0, len(dfa))):
            row = dfa.iloc[i]
            q = inpt.format(*row[['question', '1', '2', '3', '4', '5']])
            a = int(row[['correct']])
            if a == 0:
                continue
            qa = {
                "instruction": instr_template,
                "input": q,
                "correct": a,
                "output": output_template.format(a, *row[['question', str(a)]]),
            }
            w.write(json.dumps(qa, ensure_ascii=False, indent=3).strip() + "\n")


if __name__ == "__main__":
    fire.Fire(generate_data)

import json
import os
import random

import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fix_tokenizer(tokenizer):
    # Fixing broken tokenizers
    special_tokens = dict()
    for token_id in range(1000):
        token = tokenizer.convert_ids_to_tokens(token_id)
        if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad" in token:
            special_tokens["pad_token"] = token
        if tokenizer.bos_token_id in (None, tokenizer.vocab_size) and "<s>" in token:
            special_tokens["bos_token"] = token
        if tokenizer.eos_token_id in (None, tokenizer.vocab_size) and "</s>" in token:
            special_tokens["eos_token"] = token
        if tokenizer.unk_token_id in (None, tokenizer.vocab_size) and "unk" in token:
            special_tokens["unk_token"] = token
        if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep" in token:
            special_tokens["sep_token"] = token

    if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "bos_token" in special_tokens:
        special_tokens["sep_token"] = special_tokens["bos_token"]

    if tokenizer.pad_token_id in (None, tokenizer.vocab_size) and "pad_token" not in special_tokens:
        if tokenizer.unk_token_id is not None:
            special_tokens["pad_token"] = tokenizer.unk_token
        else:
            special_tokens["pad_token"] = "<|pad|>"

    if tokenizer.sep_token_id in (None, tokenizer.vocab_size) and "sep_token" not in special_tokens:
        if tokenizer.bos_token_id is not None:
            special_tokens["sep_token"] = tokenizer.bos_token
        else:
            special_tokens["sep_token"] = "<|sep|>"

    tokenizer.add_special_tokens(special_tokens)

    print("Vocab size: ", tokenizer.vocab_size)
    print("PAD: ", tokenizer.pad_token_id, tokenizer.pad_token)
    print("BOS: ", tokenizer.bos_token_id, tokenizer.bos_token)
    print("EOS: ", tokenizer.eos_token_id, tokenizer.eos_token)
    print("UNK: ", tokenizer.unk_token_id, tokenizer.unk_token)
    print("SEP: ", tokenizer.sep_token_id, tokenizer.sep_token)
    return tokenizer


def fix_model(model, tokenizer, use_resize=True):
    model.config.pad_token_id = tokenizer.pad_token_id
    assert model.config.pad_token_id is not None

    bos_candidates = (
        tokenizer.bos_token_id,
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.unk_token_id
    )
    for bos_candidate in bos_candidates:
        model.config.bos_token_id = bos_candidate
        if bos_candidate is not None:
            break
    assert model.config.bos_token_id is not None
    model.config.decoder_start_token_id = model.config.bos_token_id

    eos_candidates = (tokenizer.eos_token_id, tokenizer.sep_token_id)
    for eos_candidate in eos_candidates:
        model.config.eos_token_id = eos_candidate
        if eos_candidate is not None:
            break
    assert model.config.eos_token_id is not None

    if use_resize:
        model.resize_token_embeddings(len(tokenizer))

    return model


def gen_batch(records, batch_size):
    batch_start = 0
    while batch_start < len(records):
        batch_end = batch_start + batch_size
        batch = records[batch_start: batch_end]
        batch_start = batch_end
        yield batch


def read_jsonl(file_name):
    with open(file_name, encoding="utf-8") as r:
        return [json.loads(line) for line in r]


def convert_json_to_jsonl(input_path, output_path):
    with open(input_path, encoding="utf-8") as r, open(output_path, "w", encoding="utf-8") as w:
        data = json.load(r)
        for record in data:
            w.write(json.dumps(record, ensure_ascii=False).strip() + "\n")


def convert_for_nli(input_path: str, output_path: str):
    data = read_jsonl(input_path)

    prompt = """Учитывая посылку: 'Согласно данным ЭКГ начальным qtc 410, теперь 475., QRS 82 изначально, теперь 86, частота = 95.' и гипотезу: 'У пациента отклонения от нормы на ЭКГ', определи тип связи между этими двумя предложениями. Это entailment, contradiction или neutral?
Цепочка размышлений: В обоих предложениях говорится об ЭКГ. Значит, предложения связаны. В предложении-посылке говорится, что у пациента начальный qtc 410, далее 475, QRS 82 изначально, далее 86, частота = 95. В норме QTc 360–389 мс для мужчин и 370–399 мс – для женщин; частота в норме 50-90 ударов. Получается, у пациента удлинённый интервал QT и повышенная частота. В предложении-гипотезе указано, что у пациента отклонения от нормы ЭКГ. У пациента действительно отклонения от нормы, значит, гипотеза не противоречит посылке и связана с ней. 
Сделайте вывод: предложения связаны, гипотеза следует из предпосылки. Ответ: entailment.

Учитывая посылку: 'ИСТОРИЯ ЗАБОЛЕВАНИЯ: 86-летняя женщина с такими записями в анамнезе: диабет, гипертония, гиперхолестеринемия и обмороками. Поступившая [**2777-5-24**] с невнятной речью, затрудненным хватанием правой рукой и спутанностью сознания в течение последних двух дней.' и гипотезу: 'Неврологическая функция в норме', определи тип связи между этими двумя предложениями. Это entailment, contradiction или neutral?
Цепочка размышлений: В посылке говорится о том, что в анамнезе пациентки много заболеваний, в т.ч. обмороки. Потом она поступила в больницу с невнятной речью и спутанностью сознания. Обмороки, невнятная речь и спутанность сознания говорят о проблемах с неврологической функцией. Получается, неврологическая функция пациента отклонена от нормы. В предложении-гипотезе говорится, что у пациента неврологическая функция в норме. У пациента есть серьёзные отклонения от нормальной нейрологической функции, она не в норме. Гипотеза противоречит посылке. 
Сделайте вывод: предложения связаны, гипотеза не следует из посылки. Ответ: contradiction.

Учитывая посылку: '{0}' и гипотезу: '{1}', определи тип связи между этими двумя предложениями. Это entailment, contradiction или neutral?
"""
    with open(output_path, "w", encoding="utf-8") as w:
        for record in data:
            ru_sentence1 = record['ru_sentence1']
            ru_sentence2 = record['ru_sentence2']
            instr = prompt.format(ru_sentence1, ru_sentence2)
            row = {
                "instruction": instr,
                "gold_label": record["gold_label"],
                "pairID": record["pairID"]
            }
            w.write(json.dumps(row, ensure_ascii=False).strip() + "\n")


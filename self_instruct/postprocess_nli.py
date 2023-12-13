import argparse
import json
import os
from typing import Optional

from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from self_instruct.utils.utils import read_jsonl


entailment_keys = [
    "entailment", "association", "dependence", "dependency", "entails", "causal", "fact", "causation", "implication",
    "Историзм", "Повтор", "Метастатическое", "Вариант 1", "Метафора", "retrospective event", "confirmation",
    "Аффиляция", "Связная", "перекрестной", "cross-reference", "антецедент",
    "взаимозависимой", "взаимозависимость", "по смыслу", '"история болезни"', "Вариативная связь",
    "Межъязыковая связь", "Взаимосвязь по смыслу", '"связное предложение"', "причинной", "неявно",
    "каузальная", "медико-историческая", "причинная", "примыкание", "вход_тип_связи_1", "логической",
    "взаимообусловленность", "Взаимная", "Межъядерная", "умеренная", "логическая", "определено",
    "очевидна", "Обобщение", '"взаимодействие"', '"возможное вмешательство"', "косвенная", "Обычная",
    "определение", "Межпредметная", "взаимная причинность", "Взаимная обусловленность", "вероятно",
    "логическая связь", "взаимодополнение", "эквивалентная", "предикативная", "предпосылка",
    "имеется", "Историческая", "история заболевания", "предположение", "эквивалентности", "a) да",
    "заключается в том", "как продолжение", "Вариант 1", "как дополнение", "последовательность",
    "связь через", "умеренная", "contextual relation", "additive information", "contextual connection",
    "confirmation", "contextual relation", "chronological sequence", "contextual information",
]
contradiction_keys = [
    "contradiction", "negation", "contradictory", "Contraposition", "opposition", "exclusion",
    "Непоследовательное", "Неустойчивая", "Неравенство", "противоречивое", "противоречие", "обратная", "О2",
    '"взаимоисключение"', "инверсия", "антагонизм", "отрицается", "антонимическая", "Невозможность",
    '"противоречие"', "содержит противоречие", "ошибка", "контрадикция", "дифференциальная",
    "взаимное исключение", "взаимоисключение", "вторая", "взаимоисключающие", "несовместимость",
    "антагонистическая", "отрицание", '"Б"', "Антитезис", "связь по смыслу", "противоречивой",
    "невозможно", "невозможность", "противоречивое", "Вариант ответа: В", "конфликт", "b) да",
    "2) да", "contextual comparison", "contrastive", "противоположным", "отрицания"
]
neutral_keys = [
    "neutral", "Межсистемная", "Не установлено", "Неуверенность", "не имеет отношения",
    "не имеют отношения", "нейтральная", "неопределенность", "не связана", "независимых",
    "отсутствует", "не является ни", '"не"', "*-*", "не указана", "не установлена", "нейтрально",
    "не определена", "нейтральными", "не имеют связи", "Неактивная", "Нет связей",
    "независимое", "нейтральное", "нельзя", "не связаны", "нейтральный", "не является",
    "невозможна", "нельзя", "Неустановленная", "неопределенный", "Не указано", "отсутствует",
    "непротиворечие", "Вероятно, нет", "нет противоречия", "независимыми", "нет никакой зависимости",
    "связь отсутствует", "Вариант ответа: нет", "нет никакой связи", "нет никакой взаимосвязи",
    "независимые предложения", "связь отсутствует", "нет никакой взаимозависимости", "нет связи",
    "не связан", "нет конфликта", '"независимые"', "Не связаны", "Взаимосвязь отсутствует",
    "не связанные", "несвязанные"
]


def postprocess(input_path: str, output_predictions_path: str, output_not_predicted: Optional[str] = None,
                plot_confusion_matrix: bool = False, skip_not_predicted: bool = True) -> None:
    """
    Model prediction parser.

    If the sample could not be parsed, it is needed to pass the samples through the model again.

    Args:
        input_path: Path to the `.jsonl` file with raw predictions
        output_predictions_path: Path to the output `.jsonl` file with class labels
        output_not_predicted: Path to the output `.jsonl` file with samples that couldn't be parsed.
                              These samples are then passed through the model again
        plot_confusion_matrix: Whether to plot confusion matrix
        skip_not_predicted: `True` if write unpredicted samples to `output_not_predicted` path.
                            `False` if assign unpredicted samples label `neutral`
    """
    data = read_jsonl(input_path)

    outputs = list()
    for record in data:
        outputs.append(record)

    prediction_keys = ["entailment", "contradiction", "neutral"]
    outputs2 = list()
    outputs_not_predicted = list()
    many_answers = 0
    for record in outputs:
        record_l = record["answer"].lower()
        is_entailment = any(key.lower() in record_l for key in entailment_keys)
        is_contradiction = any(key.lower() in record_l for key in contradiction_keys)
        is_neutral = any(key.lower() in record_l for key in neutral_keys)
        answers = [is_entailment, is_contradiction, is_neutral]

        new_record = {
            "pairID": record["pairID"],
            "gold_label": record["gold_label"],
        }
        if sum(answers) == 0 and skip_not_predicted:
            outputs_not_predicted.append(record)
            continue

        elif sum(answers) == 0 and not skip_not_predicted:
            # set random class label
            new_record["prediction"] = "neutral"

        elif sum(answers) > 1:
            many_answers += 1

            new_record["prediction"] = prediction_keys[
                [i for i in range(len(answers)) if answers[i] is True][0]]

        else:
            new_record["prediction"] = prediction_keys[[i for i in range(3) if answers[i] is True][0]]

        outputs2.append(new_record)

    print("total skipped", len(outputs_not_predicted) + many_answers)
    print("outputs_not_predicted", len(outputs_not_predicted))
    print("many_answers", many_answers)

    with open(output_predictions_path, "w", encoding="utf-8") as w:
        for l in outputs2:
            w.write(json.dumps(l, ensure_ascii=False).strip() + "\n")

    if output_not_predicted is not None:
        with open(output_not_predicted, "w", encoding="utf-8") as w:
            for l in outputs_not_predicted:
                w.write(json.dumps(l, ensure_ascii=False).strip() + "\n")

    if plot_confusion_matrix:
        y_test = list(r["gold_label"] for r in outputs2)
        predictions = list(r["prediction"] for r in outputs2)
        cm = confusion_matrix(y_test, predictions, labels=prediction_keys)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=prediction_keys)
        disp.plot()
        plt.show()


def evaluate(input_folder: str):
    files = set(os.listdir(input_folder))
    task, label_id = 'RuMedNLI', 'gold_label'
    files_with_prefixes = dict()
    # group multiple preds for 1 checkpoint (multiple `output_not_predicted` files)
    # e.g. ..ckpt1_1.jsonl, ..ckpt1_2.jsonl
    for fname in files:
        fname_prefix = fname.split(".jsonl")[0][:-1]  # remove last digit from name to get prefix
        fnames_with_prefix = [os.path.join(input_folder, fn) for fn in files if fn.startswith(fname_prefix)]
        files_with_prefixes[fname_prefix] = fnames_with_prefix

    for fname_prefix, fnames in files_with_prefixes.items():
        print(f"\nProcess {fname_prefix}*.jsonl files")
        result = list()
        for fn in fnames:
            with open(fn) as f:
                result.extend(json.loads(line) for line in list(f))

        gt = [d[label_id] for d in result]
        prediction = [d['prediction'] for d in result]
        acc = round(accuracy_score(gt, prediction)*100, 2)
        print(f"Top-1 Accuracy: {acc}%")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str,
                        default="data/RuMedNLI/llama2_7b",
                        help="Path to raw generated model answers")
    args = parser.parse_args()

    root_dir = args.root_dir
    files = set(os.listdir(f"{root_dir}/raw"))
    for n in files:
        print(f"\n\nFilename: {n}")

        skip_not_predicted = True
        # if last 5-th prediction, set a 'neutral' label for unpredicted samples
        if "5" in n:
            skip_not_predicted = False
        postprocess(f"{root_dir}/raw/{n}",  # raw prediction dir
                    f"{root_dir}/pred/{n}",  # parsed samples
                    f"{root_dir}/no_pred/{n}",  # unpredicted samples
                    skip_not_predicted=skip_not_predicted)  # whether to write unpredicted samples to the file
    evaluate(f"{root_dir}/pred/")


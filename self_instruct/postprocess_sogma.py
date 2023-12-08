import argparse
import json
import os
import re
from typing import Optional

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from self_instruct.utils.utils import read_jsonl


def postprocess(input_path: str, output_predictions_path: str, output_not_predicted: Optional[str] = None,
                plot_confusion_matrix: bool = True):
    data = read_jsonl(input_path)

    outputs = list()
    for record in data:
        outputs.append(record)

    wrong_number = 0
    skipped = 0
    outputs2 = list()
    outputs_not_predicted = list()
    for record in outputs:
        # parse number of answer at the beginning or end of a sentence
        regexp = re.search(f"(?<!№\S)\d(?![^\s.,?!])", record["answer"])

        if regexp is not None:
            prediction = int(regexp.group(0))
            if prediction > 5 or prediction == 0:
                wrong_number += 1
                outputs_not_predicted.append(record)
                continue

        else:
            skipped += 1
            outputs_not_predicted.append(record)
            continue

        record["prediction"] = prediction
        outputs2.append(record)

    print("skipped", skipped)
    print("wrong number", wrong_number)

    with open(output_predictions_path, "w", encoding="utf-8") as w:
        for l in outputs2:
            w.write(json.dumps(l, ensure_ascii=False).strip() + "\n")

    if output_not_predicted is not None:
        with open(output_not_predicted, "w", encoding="utf-8") as w:
            for l in outputs_not_predicted:
                w.write(json.dumps(l, ensure_ascii=False).strip() + "\n")

    if plot_confusion_matrix:
        y_test = list(r["correct"] for r in outputs2 if r['correct'] != 0)
        predictions = list(r["correct"] for r in outputs2 if r['correct'] != 0)
        cm = confusion_matrix(y_test, predictions)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(1, 6)))
        disp.plot()
        plt.show()


def evaluate(input_folder: str):
    files = set(os.listdir(input_folder))
    task, label_id = 'RuMedTest', 'correct'
    for fname in files:
        fname = os.path.join(input_folder, fname)
        print("fname", fname)
        with open(fname, encoding="utf-8") as f:
            result = [json.loads(line) for line in list(f)]
        gt = [d[label_id] for d in result if d['prediction'] != 0 and d['correct'] != 0]
        prediction = [d['prediction'] for d in result if d['prediction'] != 0 and d['correct'] != 0]
        acc = round(accuracy_score(gt, prediction) * 100, 2)
        print(f"Top-1 Accuracy: {acc}%")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        help="Path to raw generated model answers")
    parser.add_argument("--output_predictions_path", type=str,
                        help="Save output predictions path")
    parser.add_argument("--output_not_predicted", type=str, default=None,
                        help="Save samples that the model failed to generate")
    parser.add_argument("--plot_confusion_matrix", action="store_true")
    parser.add_argument("--input_folder", type=str,
                        help="Folder with parsed predictions")
    args = parser.parse_args()

    postprocess(args.input_path, args.output_predictions_path, args.output_not_predicted,
                args.plot_confusion_matrix)
    evaluate(args.input_folder)

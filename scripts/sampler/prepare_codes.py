from typing import Dict

import yaml
import numpy as np
import pandas as pd


def prepare_codes(cfg: Dict) -> None:
    codes = pd.read_csv(cfg["icd_codes"])
    codes.columns = [
        "Category_Code",
        "Diagnosis_Code",
        "Full_Code",
        "Abbreviated_Description",
        "Full_Description",
        "Category_Title",
    ]

    with open(cfg["top_ICD10_codes_path"]) as f:
        icd_codes = f.readlines()
    icd_codes_probs = [icd.split("\n")[0].split("\t") for icd in icd_codes][1:]
    icd_codes = [s[0] for s in icd_codes_probs]
    icd_probs = np.array([float(s[1]) for s in icd_codes_probs])
    icd_probs = icd_probs / icd_probs.sum()
    icd_codes_probs = dict(zip(icd_codes, icd_probs))

    codes_filt = codes.query("Category_Code in @icd_codes")

    codes = []
    probs = []
    for cat in codes_filt["Category_Code"].unique():
        subcats = codes_filt.query("Category_Code == @cat")["Diagnosis_Code"].values
        code_p = icd_codes_probs[cat]
        for subcat in subcats:
            codes.append(cat + "." + str(subcat) if str(subcat) != "nan" else cat)
            probs.append(code_p / len(subcats))

    codes_and_probs = pd.DataFrame(list(zip(codes, probs)), columns=["codes", "probs"])
    codes_and_probs.to_csv(cfg["icd_codes_processed"], index=False)


if __name__ == "__main__":
    with open("./scripts/sampler/cfg.yaml", "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    prepare_codes(cfg)

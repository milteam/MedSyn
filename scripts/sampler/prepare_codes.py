from typing import Dict

import yaml
import numpy as np
import pandas as pd


def prepare_codes(cfg: Dict) -> None:
    """Add child categories for top_ICD10_codes.

    Data source for icd_codes:
        https://github.com/k4m1113/ICD-10-CSV/tree/master
        https://www.cms.gov/Medicare/Coding/ICD10/2018-ICD-10-CM-and-GEMs

    Args:
        cfg (Dict): _description_
    """
    codes = pd.read_csv(cfg["icd_codes"], header=None)
    codes.columns = [
        "Category_Code",
        "Diagnosis_Code",
        "Full_Code",
        "Abbreviated_Description",
        "Full_Description",
        "Category_Title",
    ]
    codes["Primary_code"] = codes["Full_Code"].apply(lambda x: x[:3])
    codes["Secondary_code"] = codes["Full_Code"].apply(lambda x: x[3:])

    with open(cfg["top_ICD10_codes_path"]) as f:
        icd_codes = f.readlines()
    icd_codes_probs = [icd.split("\n")[0].split("\t") for icd in icd_codes][1:]
    icd_codes = [s[0] for s in icd_codes_probs]
    icd_probs = np.array([float(s[1]) for s in icd_codes_probs])
    icd_probs = icd_probs / icd_probs.sum()
    icd_codes_probs = dict(zip(icd_codes, icd_probs))

    codes_filt = codes.query("Primary_code in @icd_codes")

    mask = codes_filt["Secondary_code"].apply(lambda x: len(x)) <= cfg["max_icd_depth"]
    codes_filt = codes_filt[mask]

    codes = []
    probs = []
    for cat in codes_filt["Primary_code"].unique():
        subcats = codes_filt.query("Primary_code == @cat")["Secondary_code"].values
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

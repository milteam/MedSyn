from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


class ICDCodeSampler:
    def __init__(
        self,
        path_to_dis: str,
        icd_gender_split: Dict,
        gender_scpec_or_general: List[float],
    ) -> None:
        self.icd_gender_split = icd_gender_split
        self.load_dis(path_to_dis)
        self.gender_scpec_or_general = gender_scpec_or_general
        self.get_gender_split()

    def load_dis(self, path: str) -> None:
        self.icd_codes_probs = pd.read_csv(path)
        self.icd_codes_probs = dict(
            zip(
                self.icd_codes_probs["codes"].values,
                self.icd_codes_probs["probs"].values,
            )
        )
        self.icd_codes = list(self.icd_codes_probs.keys())

    def get_gender_split(self) -> Tuple[List, List]:
        male = self.icd_gender_split[
            "List of categories limited to, or more likely to occur in, male persons"
        ]
        female = self.icd_gender_split[
            "List of categories limited to, or more likely to occur in, female persons"
        ]
        self.icd_codes_male = [code for code in self.icd_codes if code in male]
        self.icd_codes_female = [code for code in self.icd_codes if code in female]
        self.icd_codes_neutral = [
            code
            for code in self.icd_codes
            if code not in self.icd_codes_male + self.icd_codes_female
        ]

        self.icd_codes_male_p = np.array(
            [self.icd_codes_probs[code] for code in self.icd_codes_male]
        )
        self.icd_codes_female_p = np.array(
            [self.icd_codes_probs[code] for code in self.icd_codes_female]
        )
        self.icd_codes_neutral_p = np.array(
            [self.icd_codes_probs[code] for code in self.icd_codes_neutral]
        )

        # Renormalize:
        self.icd_codes_male_p = self.icd_codes_male_p / self.icd_codes_male_p.sum()
        self.icd_codes_female_p = (
            self.icd_codes_female_p / self.icd_codes_female_p.sum()
        )
        self.icd_codes_neutral_p = (
            self.icd_codes_neutral_p / self.icd_codes_neutral_p.sum()
        )

    def get_sample(self, gender: str) -> str:
        case_ = np.random.choice(
            ["specific", "general"], p=self.gender_scpec_or_general
        )

        if case_ == "general":
            return np.random.choice(self.icd_codes_neutral, p=self.icd_codes_neutral_p)
        if gender == "male":
            sample = np.random.choice(self.icd_codes_male, p=self.icd_codes_male_p)
        elif gender == "female":
            sample = np.random.choice(self.icd_codes_female, p=self.icd_codes_female_p)
        return sample


class Disease:
    def __init__(self, disease_db_path: str) -> None:
        self.diseases = pd.read_csv(disease_db_path)
        self.prepare_df()

    def prepare_df(self) -> None:
        self.diseases = self.diseases.dropna(subset=["icd_10", "icd_9"])
        self.diseases["icd_10_p"] = self.diseases["icd_10"].apply(
            lambda x: x.split(".")[0]
        )

    def get_data_by_icd(self, icd_code: str) -> Dict:
        data = self.diseases.query("icd_10 == @icd_code")

        disease_info = {}
        if data.shape[0] == 0:
            return disease_info

        for idx in range(data.shape[0]):
            disease_info[data.iloc[idx]["name"]] = {
                "Name": data.iloc[idx]["name"],
                "DOID": data.iloc[idx]["do_id"],
                "ICD_10": data.iloc[idx]["icd_10"],
                "MESH_ID": data.iloc[idx]["mesh_id"],
            }

        return disease_info
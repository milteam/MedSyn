from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


class DiseaseSampler:
    def __init__(self, cfg: Dict) -> None:
        self.diseases = pd.read_csv(cfg["main_df_diseases_data"])
        self.prepare_df()
        self.gender_scpec_or_general = cfg["gender_scpec_or_general"]
        self.prepare_gender_probs()
        self.cfg = cfg

    def prepare_df(self) -> None:
        self.diseases = self.diseases.dropna(subset=["symptoms"], axis=0)

    def prepare_gender_probs(self) -> None:
        self.icd_codes_male = self.diseases.query("gender == 'male'")["icd_10"]
        self.icd_codes_female = self.diseases.query("gender == 'female'")["icd_10"]
        self.icd_codes_neutral = self.diseases.query("gender == 'neutral'")["icd_10"]

        self.icd_codes_male_p = self.diseases.query("gender == 'male'")["probs"]
        self.icd_codes_female_p = self.diseases.query("gender == 'female'")["probs"]
        self.icd_codes_neutral_p = self.diseases.query("gender == 'neutral'")["probs"]

        # Renormalize:
        self.icd_codes_male_p = self.icd_codes_male_p / self.icd_codes_male_p.sum()
        self.icd_codes_female_p = (
            self.icd_codes_female_p / self.icd_codes_female_p.sum()
        )
        self.icd_codes_neutral_p = (
            self.icd_codes_neutral_p / self.icd_codes_neutral_p.sum()
        )

    def softmax(self, values: np.ndarray) -> np.ndarray:
        T = self.cfg["symptoms_temperature"]
        p = np.exp(values / T) / np.sum(np.exp(values / T))
        if np.isnan(p).any():
            p = np.array([1 / len(values) for _ in range(len(values))])
        return p

    def sample_icd(self, gender: str):
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

    def sample_symptoms(self, data: pd.DataFrame) -> Dict:
        symptoms_name = data["symptoms"].values[0]
        symptoms_tfidf = data["sympotoms_tf_idfs"].values[0]
        symptoms_name = [
            s_name.replace(" '", "").replace("'", "")
            for s_name in symptoms_name.replace("[", "").replace("]", "").split("',")
        ]
        n_symptoms = min(
            len(symptoms_name), np.random.randint(1, self.cfg["max_symptoms"])
        )
        symptoms_tfidf = data["sympotoms_tf_idfs"].values[0]
        symptoms_tfidf = np.array(
            [
                float(tf_idf)
                for tf_idf in symptoms_tfidf.replace("[", "")
                .replace("]", "")
                .split(",")
            ]
        )
        symptoms_scores = self.softmax(symptoms_tfidf)
        symptoms = np.random.choice(
            symptoms_name, p=symptoms_scores, size=n_symptoms, replace=False
        )
        return symptoms

    def get_sample(self, gender: str) -> Dict:
        icd_code = self.sample_icd(gender)
        data = self.diseases.query("icd_10 == @icd_code")
        doid = data["DOID"].values[0]
        mesh_id = data["MESH"].values[0]
        symptoms = self.sample_symptoms(data)

        disease_info = {
            "disease": {
                "icd_code": icd_code,
                "DOID": doid,
                "MESH_ID": mesh_id,
                "name": data["Name"].values[0],
                "descriptions": data["Descriptions"].values[0],
            },
            "symptoms": list(symptoms),
        }

        return disease_info

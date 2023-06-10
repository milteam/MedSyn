from typing import Dict, Tuple, List

import json
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
import warnings

warnings.simplefilter(action="ignore")


class DemographicSampler:
    def __init__(
        self,
        edge_end_gender_path: str,
        family_status_path: str,
        ethnic_groups_path: str,
        min_age: int,
    ) -> None:
        self.edge_gender = pd.read_csv(edge_end_gender_path)
        self.family_status = pd.read_csv(family_status_path)
        self.ethnic_groups = pd.read_csv(ethnic_groups_path)
        self.merige_temp = 1.25  # Heuristic to increase merrige prob
        self.calc_stats()
        self.min_age = min_age

    def prepare_ethic_groups(self):
        self.ethnic_groups["Процент от населения"] = self.ethnic_groups[
            "Процент от населения"
        ].apply(lambda x: float(x.split(" ")[0].replace(",", ".")) / 100)
        self.ethnic_groups["Процент от населения"] = (
            self.ethnic_groups["Процент от населения"]
            / self.ethnic_groups["Процент от населения"].sum()
        )

        self.groups = self.ethnic_groups["Народность"]
        self.groups_p = self.ethnic_groups["Процент от населения"]

    def prepare_family_status(self):
        latels_year = self.family_status.year.max()
        latelst = self.family_status.query("year == @latels_year")

        # Move to cumsum as data represent number of new marriges per age group:
        latelst["m_age_18-24"] = latelst["m_age_0-17"] + latelst["m_age_18-24"]
        latelst["m_age_25-34"] = latelst["m_age_18-24"] + latelst["m_age_25-34"]
        latelst["m_age_35-99"] = latelst["m_age_25-34"] + latelst["m_age_35-99"]

        latelst["f_age_18-24"] = latelst["f_age_0-17"] + latelst["f_age_18-24"]
        latelst["f_age_25-34"] = latelst["f_age_18-24"] + latelst["f_age_25-34"]
        latelst["f_age_35-99"] = latelst["f_age_25-34"] + latelst["f_age_35-99"]

        mer_m_total = (
            latelst[["m_age_0-17", "m_age_18-24", "m_age_25-34", "m_age_35-99", "m_na"]]
            .sum()
            .sum()
        )
        mer_f_total = (
            latelst[["f_age_0-17", "f_age_18-24", "f_age_25-34", "f_age_35-99", "f_na"]]
            .sum()
            .sum()
        )

        # NOTE: this is just approximation.
        # More accurate estimation should compute total of meriges - divorces and account full population.
        self.marrige_p_m = {
            "0-17": 0,
            "18-24": latelst["m_age_18-24"].values[0] / mer_m_total * self.merige_temp,
            "25-34": latelst["m_age_25-34"].values[0] / mer_m_total * self.merige_temp,
            "35-99": latelst["m_age_35-99"].values[0] / mer_m_total * self.merige_temp,
        }

        self.marrige_p_f = {
            "0-17": 0,
            "18-24": latelst["f_age_18-24"].values[0] / mer_f_total * self.merige_temp,
            "25-34": latelst["f_age_25-34"].values[0] / mer_f_total * self.merige_temp,
            "35-99": latelst["f_age_35-99"].values[0] / mer_f_total * self.merige_temp,
        }

    def calc_stats(self):
        city_m_total = self.edge_gender["M_City"].sum()
        city_f_total = self.edge_gender["F_City"].sum()
        coun_m_total = self.edge_gender["M_Countryside"].sum()
        coun_f_total = self.edge_gender["F_Countryside"].sum()

        total_city = city_m_total + city_f_total
        total_coun = coun_m_total + coun_f_total

        total_popu = total_city + total_coun

        self.location = ["city", "countyside"]
        self.p_location = [total_city / total_popu, total_coun / total_popu]

        self.gender = ["male", "female"]
        self.p_gender_city = [city_m_total / total_city, city_f_total / total_city]
        self.p_gender_coun = [coun_m_total / total_coun, coun_f_total / total_coun]

        self.ages = self.edge_gender["Age"]
        self.p_age_m_city = self.edge_gender["M_City"] / city_m_total
        self.p_age_f_city = self.edge_gender["F_City"] / city_f_total
        self.p_age_m_coun = self.edge_gender["M_Countryside"] / coun_m_total
        self.p_age_f_coun = self.edge_gender["F_Countryside"] / coun_f_total

        self.prepare_family_status()
        self.prepare_ethic_groups()

    def sample_ethical_group(self):
        return np.random.choice(self.groups, p=self.groups_p)

    def sample_location(self):
        return np.random.choice(self.location, p=self.p_location)

    def sample_gender(self, location):
        if location == "city":
            return np.random.choice(self.gender, p=self.p_gender_city)
        elif location == "countyside":
            return np.random.choice(self.gender, p=self.p_gender_coun)

    def sample_age(self, location, gender):
        if location == "city":
            if gender == "male":
                return np.random.choice(self.ages, p=self.p_age_m_city)
            elif gender == "female":
                return np.random.choice(self.ages, p=self.p_age_f_city)
        elif location == "countyside":
            if gender == "male":
                return np.random.choice(self.ages, p=self.p_age_m_coun)
            elif gender == "female":
                return np.random.choice(self.ages, p=self.p_age_f_coun)

    def sample_family_state(self, gender, age):
        if age < 18:
            key = "0-17"
        elif 18 <= age <= 24:
            key = "18-24"
        elif 25 <= age <= 34:
            key = "25-34"
        elif 35 <= age <= 99:
            key = "35-99"

        if gender == "male":
            p = self.marrige_p_m[key]

        elif gender == "female":
            p = self.marrige_p_f[key]

        return np.random.choice([1, 0], p=[p, 1 - p])

    def get_sample(self):
        """
        P(gender, age, location) = P(age | gender, location) * P(gender | location) * P(location)
        """
        location = self.sample_location()
        ethical_group = self.sample_ethical_group()
        gender = self.sample_gender(location)
        age = self.sample_age(location, gender)
        while age < self.min_age:
            age = self.sample_age(location, gender)

        family_state = self.sample_family_state(gender, age)

        return {
            "ethical_group": ethical_group,
            "location": location,
            "gender": gender,
            "age": int(age),
            "family_state": int(family_state),
        }


class ICDCodeSampler:
    def __init__(
        self,
        path_to_dis: str,
        icd_gender_split: Dict,
        gender_scpec_or_general: List[float],
    ) -> None:
        self.load_dis(path_to_dis)
        self.icd_gender_split = icd_gender_split
        self.get_gender_split()
        self.gender_scpec_or_general = gender_scpec_or_general

    def load_dis(self, path: str) -> None:
        with open(path) as f:
            icd_codes = f.readlines()
        self.icd_codes = [icd.split("\n")[0] for icd in icd_codes]

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

    def get_sample(self, gender: str) -> str:
        case_ = np.random.choice(
            ["specific", "general"], p=self.gender_scpec_or_general
        )

        if case_ == "general":
            return np.random.choice(self.icd_codes_neutral)
        if gender == "male":
            sample = np.random.choice(self.icd_codes_male)
        elif gender == "female":
            sample = np.random.choice(self.icd_codes_female)
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
        data = self.diseases.query("icd_10_p == @icd_code")

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


class Symptoms:
    def __init__(self, disease_symptom_rel_path: str, symptom_mesh_path: str) -> None:
        self.disease_symptom_rel = pd.read_csv(disease_symptom_rel_path)
        self.symptom_mesh = pd.read_csv(symptom_mesh_path, sep="\t")
        self.prepare_df()

    def prepare_df(self):
        clmns = [clmn.replace(" ", "_") for clmn in self.symptom_mesh.columns]
        self.symptom_mesh.columns = clmns

    def get_symptoms(self, disease: Dict) -> Dict:
        do_id = disease["DOID"]
        mesh_id = disease["MESH_ID"]
        symptoms = self.disease_symptom_rel.query("Disease == @do_id")["Symptom"].values

        symptoms_data = {}
        for symptom in symptoms:
            s_id = symptom.split(":")[-1]
            symptom_term = self.symptom_mesh.query("MeSH_Symptom_ID == @s_id")
            symptom_term = symptom_term.query("MeSH_Disease_ID == @mesh_id")

            for idx in range(symptom_term.shape[0]):
                symptoms_data[symptom] = {
                    "Term": symptom_term.iloc[idx]["MeSH_Symptom_Term"],
                    "TFIDF": symptom_term.iloc[idx]["TFIDF_score"],
                }

        return symptoms_data


class Sampler:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.set_seed(cfg["seed"])
        gender_split = self.load_gender_split()
        self.demographic_sampler = DemographicSampler(
            cfg["edge_gender_pyramyd_path"],
            cfg["family_status_path"],
            cfg["ethnic_groups_path"],
            cfg["min_age"],
        )
        self.icd_sampler = ICDCodeSampler(
            cfg["top_ICD10_codes_path"],
            gender_split,
            cfg["gender_scpec_or_general"]
        )
        self.disease_sampler = Disease(cfg["disease_path"])
        self.symptoms_sampler = Symptoms(
            cfg["disease_symptom_relation_path"], cfg["mesh_disease_symp_path"]
        )

    @staticmethod
    def set_seed(seed) -> None:
        np.random.seed(seed)
        random.seed(seed)

    def softmax(self, values: np.ndarray) -> np.ndarray:
        T = self.cfg["symptoms_temperature"]
        return np.exp(values / T) / np.sum(np.exp(values / T))

    def load_gender_split(self) -> Dict:
        with open(self.cfg["icd_gender_split_path"], "r") as f:
            icd_gender_split_path = json.load(f)
            return icd_gender_split_path

    def get_data_for_icd(self, icd_code: str) -> Dict:
        data = {}
        disease_sample = self.disease_sampler.get_data_by_icd(icd_code)
        if disease_sample:
            for disease_name, disease_data in disease_sample.items():
                symptoms = self.symptoms_sampler.get_symptoms(disease_data)
                if symptoms:
                    n_symptoms = min(
                        len(symptoms), np.random.randint(1, self.cfg["max_symptoms"])
                    )
                    symptoms_name = [symp["Term"] for symp in symptoms.values()]
                    symptoms_tfidf = np.array(
                        [symp["TFIDF"] for symp in symptoms.values()]
                    )
                    symptoms_scores = self.softmax(symptoms_tfidf)
                    symptoms = np.random.choice(
                        symptoms_name, p=symptoms_scores, size=n_symptoms, replace=False
                    )

                    data = {
                        "icd_code": icd_code,
                        "disease": disease_data,
                        "symptoms": symptoms,
                    }
        return data

    def get_sample(self) -> Dict:
        sample = self.demographic_sampler.get_sample()
        sample["disease"] = []
        sample["symptoms"] = []

        # sample = {"gender": gender, "age": age, "disease": [], "symptoms": []}

        n_diseases = np.random.randint(1, self.cfg["max_diseases"] + 1)
        for _ in range(n_diseases):
            disease_icd = self.icd_sampler.get_sample(sample["gender"])
            sample_ = self.get_data_for_icd(disease_icd)
            if sample_:
                disease_info = sample_["disease"]
                disease_info["idc_10"] = sample_["icd_code"]

                # Add only if disease is new:
                if sample["disease"]:
                    if sample_["disease"]["Name"] not in [
                        d["Name"] for d in sample["disease"]
                    ]:
                        sample["disease"].append(disease_info)
                        sample["symptoms"].extend(sample_["symptoms"])
                else:
                    sample["disease"].append(disease_info)
                    sample["symptoms"].extend(sample_["symptoms"])

        return sample

    def generate_samples(self) -> List[Dict]:
        samples = {}
        idx = 0
        for _ in tqdm(range(self.cfg["max_iters"])):
            sample = self.get_sample()
            if sample["disease"]:
                samples[idx] = sample
                idx += 1

            if idx > self.cfg["n_samples"]:
                print(f"Generation of {idx - 1} samples is done.")
                return samples

        print(f"Max iterations reached, number of generated samples: {idx - 1}.")

from typing import Dict, Tuple, List

import json
import random
import shortuuid
import numpy as np
import pandas as pd

from tqdm import tqdm
import warnings

from drug_sampler import DrugSampler
from disease_sampler import DiseaseSampler

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


class SmokingSampler:
    def __init__(self) -> None:
        male_p = 34.6 / 100
        female_p = 8.3 / 100
        self.male_p = [1 - male_p, male_p]
        self.female_p = [1 - female_p, female_p]
        self.age_threshold = 16

    def get_sample(self, sample: Dict) -> int:
        gender = sample["gender"]
        age = sample["age"]
        if age < self.age_threshold:
            return 0
        if gender == "male":
            return np.random.choice([0, 1], p=self.male_p)
        if gender == "female":
            return np.random.choice([0, 1], p=self.female_p)


class Sampler:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg
        self.set_seed(cfg["seed"])

        self.demographic_sampler = DemographicSampler(
            cfg["edge_gender_pyramyd_path"],
            cfg["family_status_path"],
            cfg["ethnic_groups_path"],
            cfg["min_age"],
        )
        self.disease_sampler = DiseaseSampler(cfg)
        self.drug_sampler = DrugSampler(
            cfg["drug_disease_rel_path"],
            cfg["drugbank_path"],
            cfg["disease_to_disease_path"],
            cfg["indications_path"],
            cfg["meddra_freq_path"],
            cfg["max_side_effects"],
        )
        self.smoking_sampler = SmokingSampler()

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

    def sample_drug_and_se(self, disease_doid: str):
        return self.drug_sampler.get_sample(disease_doid)

    def get_sample(self) -> Dict:
        sample = self.demographic_sampler.get_sample()
        smoke = self.smoking_sampler.get_sample(sample)
        sample["smoking"] = int(smoke)
        sample["disease"] = []
        sample["symptoms"] = []

        n_diseases = np.random.randint(1, self.cfg["max_diseases"] + 1)
        for _ in range(n_diseases):
            sample_ = self.disease_sampler.get_sample(sample["gender"])
            disease_info = sample_["disease"]
            disease_info["idc_10"] = disease_info["icd_code"]

            # Add only if disease is new:
            if sample["disease"]:
                if sample_["disease"]["name"] not in [
                    d["name"] for d in sample["disease"]
                ]:
                    sample["disease"].append(disease_info)
                    sample["symptoms"].extend(sample_["symptoms"])
            else:
                sample["disease"].append(disease_info)
                sample["symptoms"].extend(sample_["symptoms"])

        if sample["disease"]:
            do_ids = [
                data["DOID"] for data in sample["disease"] if data["DOID"] is not np.nan
            ]
            if do_ids:
                do_id = np.random.choice(do_ids)
                drug_and_se = self.sample_drug_and_se(do_id)
                sample["drug_and_se"] = drug_and_se

        return sample

    def generate_samples(self) -> List[Dict]:
        samples = {}
        idx = 0
        for _ in tqdm(range(self.cfg["max_iters"])):
            sample = self.get_sample()
            if sample["disease"]:
                samples[idx] = sample
                UID = shortuuid.uuid()
                samples["UID"] = UID
                idx += 1

            if idx > self.cfg["n_samples"]:
                print(f"Generation of {idx - 1} samples is done.")
                return samples

        print(f"Max iterations reached, number of generated samples: {idx - 1}.")

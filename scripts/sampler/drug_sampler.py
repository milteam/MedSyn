import numpy as np
import pandas as pd


class DrugSampler:
    """
    Sample drug relevant to disease and side effects for the drug.
    Disease may be changed on similar disease.
    Disease defined by DOID.
    """

    def __init__(
        self,
        drug_disease_rel_path: str,
        drugbank_path: str,
        disease_to_disease_path: str,
        indications_path: str,
        meddra_freq_path: str,
        max_side_effects: int,
    ) -> None:
        self.drug_disease_rel = pd.read_csv(drug_disease_rel_path)
        self.relations = self.drug_disease_rel.columns[2:-2]

        self.drugbank = pd.read_csv(drugbank_path, on_bad_lines="skip")
        self.drugbank.columns = ["name", "db_idx"]

        self.disease_to_disease = pd.read_csv(disease_to_disease_path)
        self.disease_to_disease = self.disease_to_disease.query("Resemble == 1")

        self.indications = pd.read_csv(indications_path)
        self.meddra_freq = pd.read_csv(meddra_freq_path)
        self.meddra_freq = self.meddra_freq.query("placebo != placebo")
        self.meddra_freq["mean_freq"] = (
            self.meddra_freq["lower"] + self.meddra_freq["lower"]
        ) / 2
        self.max_side_effects = max_side_effects

    def sample_similar_disease(self, disease_doid: str):
        data = self.disease_to_disease.query("Disease_1 == @disease_doid")
        if data.shape[0] > 0:
            disease_doid = data.sample(1)["Disease_2"].values[0]
        else:
            data = self.disease_to_disease.query("Disease_2 == @disease_doid")
            if data.shape[0] > 0:
                disease_doid = data.sample(1)["Disease_1"].values[0]
        return disease_doid

    def sample_drug(self, disease_doid: str):
        # Switch to similar diseases for more diversity:
        # TODO: add probability for this sampling:
        drug_data = {}
        if np.random.rand() < 0.5:
            disease_doid = self.sample_similar_disease(disease_doid)

        disease_data = self.drug_disease_rel.query(
            "Disease == @disease_doid & Inferred_Relation == 0"
        )
        if disease_data.shape[0] > 0:
            sample = disease_data.sample(1)
            relation = self.relations[np.where(sample.values[:, 2:-2][0] == 1)[0][0]]
            drug_db = sample["Drug"].values[0].split(":")[-1]
            drug_name = self.drugbank.query("db_idx == @drug_db")["name"]
            if drug_name.shape[0] > 0:
                drug_name = drug_name.values[0]

                drug_data = {
                    "disease_DOID": disease_doid,
                    "drug_db_idx": drug_db,
                    "drug_name": drug_name,
                    "drug_to_disease_relation": relation,
                }

        return drug_data

    def sample_side_effect(self, drugbank_id: str):
        side_effects = []

        stitch_id_flat = self.indications.query("drugbank_id == @drugbank_id")[
            "stitch_id_flat"
        ].tolist()
        if stitch_id_flat:
            stitch_id_flat = stitch_id_flat[0]
        if stitch_id_flat:
            data = self.meddra_freq.query("stitch_id_flat == @stitch_id_flat")
            freq = data["mean_freq"].tolist()
            se = data["side_effect_name"].tolist()

            for f, se in zip(freq, se):
                if se not in side_effects:
                    if np.random.rand() < f:
                        side_effects.append(se)
            if len(side_effects) > self.max_side_effects:
                side_effects = np.random.choice(
                    side_effects, size=self.max_side_effects
                ).tolist()

        return side_effects

    def get_sample(self, disease_doid: str):
        sample = {}
        drug = self.sample_drug(disease_doid)
        if drug:
            sample = {"initial_disease_DOID": disease_doid}
            sample.update(drug)
            side_effect = self.sample_side_effect(drug["drug_db_idx"])
            sample["side_effect"] = side_effect

        return sample

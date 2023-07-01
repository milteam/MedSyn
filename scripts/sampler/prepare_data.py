from typing import Dict

import json
import yaml
import numpy as np
import pandas as pd


def add_mesh_codes(
    main_df: pd.DataFrame, mesh2icd10_mapping: pd.DataFrame
) -> pd.DataFrame:
    mesh_codes = []
    ulms_codes = []
    descriptions = []

    for i in range(len(main_df)):
        icd10 = main_df.loc[i]["icd_10"]
        mesh = [main_df.loc[i]["MESH"]]
        ulms = [main_df.loc[i]["ULMS"]]
        desc = [np.nan]

        map_data = mesh2icd10_mapping.query("ICD10CM_CODE == @icd10")
        if map_data.shape[0] > 0:
            mesh_ = ["MESH:" + id for id in map_data["MESH_ID"].values]
            ulms_ = ["UMLS_CUI:" + id for id in map_data["UMLS_CUI"].values]
            desc = map_data["DRESCP"].values.tolist()

            if mesh[0] is not np.nan:
                if not mesh[0] in mesh_:
                    mesh.extend(mesh_)
            else:
                mesh = mesh_

            if ulms[0] is not np.nan:
                if ulms[0] not in ulms_:
                    ulms.extend(ulms_)
            else:
                ulms = ulms_

        mesh_codes.append(mesh if mesh[0] is not np.nan else np.nan)
        ulms_codes.append(ulms if ulms[0] is not np.nan else np.nan)
        descriptions.append(desc if desc[0] is not np.nan else np.nan)

    main_df["MESH"] = mesh_codes
    main_df["ULMS"] = ulms_codes
    main_df["Descriptions"] = descriptions

    return main_df


def add_probabilities(
    main_df: pd.DataFrame, codes_and_probs: pd.DataFrame
) -> pd.DataFrame:
    main_df = pd.merge(main_df, codes_and_probs, on="icd_10", how="left")
    return main_df


def add_gender(main_df: pd.DataFrame, gender_split: Dict) -> pd.DataFrame:
    female = gender_split[
        "List of categories limited to, or more likely to occur in, female persons"
    ]
    male = gender_split[
        "List of categories limited to, or more likely to occur in, male persons"
    ]
    gender_attr = []
    for i in range(len(main_df)):
        code = main_df.loc[i]["icd_10"]
        if code in male:
            attr = "male"
        elif code in female:
            attr = "female"
        else:
            attr = "neutral"
        gender_attr.append(attr)

    assert len(gender_attr) == len(main_df)

    main_df["gender"] = gender_attr

    return main_df


def add_diseases(main_df: pd.DataFrame, disease_df: pd.DataFrame) -> pd.DataFrame:
    main_df_codes = main_df["icd_10"].values
    disease_df.dropna(axis=0, subset=["icd_10"], inplace=True)
    disease_df = disease_df.query("icd_10 in @main_df_codes")

    mesh_codes = []
    doid_codes = []
    pharmgkb_ids = []

    for i in range(len(main_df)):
        icd_code = main_df.loc[i]["icd_10"]
        mesh_code = main_df.loc[i]["MESH"]
        doid_code = [main_df.loc[i]["DOID"]]

        if mesh_code is np.nan:
            mesh_code_d = disease_df.query("icd_10 == @icd_code")["mesh_id"].values
            mesh_code = ["MESH:" + code for code in mesh_code_d if code is not np.nan]
            if not mesh_code:
                mesh_code = np.nan

        if doid_code is np.nan:
            doid_code_d = disease_df.query("icd_10 == @icd_code")["do_id"].values
            doid_code = [code for code in doid_code_d if code is not np.nan]
            if not doid_code:
                doid_code = np.nan

        pharmgkb_id = np.nan
        pharmgkb_d = disease_df.query("icd_10 == @icd_code")["pharmgkb_id"].values
        if pharmgkb_d.shape[0] > 0:
            pharmgkb_id = pharmgkb_d[0]

        mesh_codes.append(mesh_code)
        doid_codes.append(doid_code[0])
        pharmgkb_ids.append(pharmgkb_id)

    main_df["DOID"] = doid_codes
    main_df["MESH"] = mesh_codes
    main_df["pharmgkb_ids"] = pharmgkb_ids

    return main_df


def add_symptoms(
    main_df: pd.DataFrame, disease_symptom_rel: pd.DataFrame, symptom_mesh: pd.DataFrame
) -> pd.DataFrame:
    clmns = [clmn.replace(" ", "_") for clmn in symptom_mesh.columns]
    symptom_mesh.columns = clmns
    symptom_mesh["MeSH_Disease_ID"] = symptom_mesh["MeSH_Disease_ID"].apply(
        lambda x: "MESH:" + x if x is not np.nan else np.nan
    )
    symptom_mesh["MeSH_Symptom_ID"] = symptom_mesh["MeSH_Symptom_ID"].apply(
        lambda x: "MESH:" + x if x is not np.nan else np.nan
    )

    symptoms = []
    symptoms_mesh_ids = []
    sympotoms_tf_idfs = []

    for i in range(len(main_df)):
        doid = main_df.loc[i]["DOID"]
        mesh_ids = main_df.loc[i]["MESH"]

        symp_ids = []
        sympt_str = []
        symp_tf_idf = []

        if doid is not np.nan and mesh_ids is not np.nan:
            mesh_sympt_id = disease_symptom_rel.query("Disease == @doid")[
                "Symptom"
            ].values

            for symp in mesh_sympt_id:
                symp_ids.append(symp)

                symptom_term = symptom_mesh.query("MeSH_Symptom_ID == @symp")
                symptom_term = symptom_term.query("MeSH_Disease_ID in @mesh_ids")

                for idx in range(symptom_term.shape[0]):
                    sympt_str.append(symptom_term.iloc[idx]["MeSH_Symptom_Term"])
                    symp_tf_idf.append(symptom_term.iloc[idx]["TFIDF_score"])

        if not symp_ids:
            if doid is not np.nan and mesh_ids is np.nan:
                mesh_sympt_id = disease_symptom_rel.query("Disease == @doid")[
                    "Symptom"
                ].values

                for symp in mesh_sympt_id:
                    symp_ids.append(symp)

                    symptom_term = symptom_mesh.query("MeSH_Symptom_ID == @symp")

                    for idx in range(symptom_term.shape[0]):
                        sympt_str.append(symptom_term.iloc[idx]["MeSH_Symptom_Term"])
                        symp_tf_idf.append(1)

            if doid is np.nan and mesh_ids is not np.nan:
                for mesh_id in mesh_ids:
                    symptom_term = symptom_mesh.query("MeSH_Disease_ID == @mesh_id")

                    for idx in range(symptom_term.shape[0]):
                        symp_ids.append(symptom_term.iloc[idx]["MeSH_Symptom_ID"])
                        sympt_str.append(symptom_term.iloc[idx]["MeSH_Symptom_Term"])
                        symp_tf_idf.append(symptom_term.iloc[idx]["TFIDF_score"])

        if sympt_str:
            assert len(sympt_str) == len(symp_tf_idf)

        symptoms_mesh_ids.append(symp_ids if symp_ids else np.nan)
        symptoms.append(sympt_str if sympt_str else np.nan)
        sympotoms_tf_idfs.append(symp_tf_idf if symp_tf_idf else np.nan)

    main_df["symptoms"] = symptoms
    main_df["symptoms_mesh_id"] = symptoms_mesh_ids
    main_df["sympotoms_tf_idfs"] = sympotoms_tf_idfs

    return main_df


def add_symptoms_do(
    main_df: pd.DataFrame, symptom_DO: pd.DataFrame, symptom_mesh: pd.DataFrame
) -> pd.DataFrame:
    symptom_DO["disease_id"] = symptom_DO["disease_id"].apply(
        lambda x: "MESH:" + x if x is not np.nan else np.nan
    )
    symptom_DO["symptom_id"] = symptom_DO["symptom_id"].apply(
        lambda x: "MESH:" + x if x is not np.nan else np.nan
    )

    symptoms = []
    symptoms_mesh_ids = []
    sympotoms_tf_idfs = []

    for i in range(len(main_df)):
        doid = main_df.loc[i]["DOID"]
        mesh_ids = main_df.loc[i]["MESH"]

        symptoms_ = main_df.loc[i]["symptoms"]
        symptoms_ids = main_df.loc[i]["symptoms_mesh_id"]
        symptoms_tfidfs = main_df.loc[i]["sympotoms_tf_idfs"]

        if symptoms_ is np.nan:
            do_data = symptom_DO.query("doid_code == @doid")
            if do_data.shape[0] > 0:
                symptoms_ = []
                symptoms_ids = []
                symptoms_tfidfs = []
                for i in range(do_data.shape[0]):
                    symptoms_.append(do_data.iloc[i]["symptom_name"])
                    symptoms_ids.append(do_data.iloc[i]["symptom_id"])
                    symptoms_tfidfs.append(do_data.iloc[i]["tfidf_score"])
            else:
                if mesh_ids is not np.nan:
                    do_data = symptom_DO.query("disease_id in @mesh_ids")
                    if do_data.shape[0] > 0:
                        symptoms_ = []
                        symptoms_ids = []
                        symptoms_tfidfs = []
                        for i in range(do_data.shape[0]):
                            symptoms_.append(do_data.iloc[i]["symptom_name"])
                            symptoms_ids.append(do_data.iloc[i]["symptom_id"])
                            symptoms_tfidfs.append(do_data.iloc[i]["tfidf_score"])

        if symptoms_ is not np.nan:
            assert len(symptoms_) == len(symptoms_tfidfs)

        symptoms.append(symptoms_)
        symptoms_mesh_ids.append(symptoms_ids)
        sympotoms_tf_idfs.append(symptoms_tfidfs)

    main_df["symptoms"] = symptoms
    main_df["symptoms_mesh_id"] = symptoms_mesh_ids
    main_df["sympotoms_tf_idfs"] = sympotoms_tf_idfs

    return main_df


def prepare_data(cfg: Dict) -> None:
    main_df = pd.read_csv(cfg["main_df"])
    mesh2icd10_mapping = pd.read_csv(cfg["mesh2icd"])
    codes_and_probs = pd.read_csv(cfg["icd_codes_processed"])
    with open(cfg["icd_gender_split_path"], "r") as f:
        gender_split = json.load(f)

    disease_df = pd.read_csv(cfg["disease_path"])
    disease_symptom_rel = pd.read_csv(cfg["disease_symptom_relation_path"])
    symptom_mesh = pd.read_csv(cfg["mesh_disease_symp_path"], sep="\t")
    symptom_DO = pd.read_csv(cfg["symptoms_DO"], sep="\t")

    main_df = add_mesh_codes(main_df, mesh2icd10_mapping)
    main_df = add_probabilities(main_df, codes_and_probs)
    main_df = add_gender(main_df, gender_split)
    main_df = add_diseases(main_df, disease_df)
    main_df = add_symptoms(main_df, disease_symptom_rel, symptom_mesh)
    main_df = add_symptoms_do(main_df, symptom_DO, symptom_mesh)

    main_df.to_csv("./data/main_proc.csv", index=False)


if __name__ == "__main__":
    with open("./scripts/sampler/cfg.yaml", "rb") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    prepare_data(cfg)

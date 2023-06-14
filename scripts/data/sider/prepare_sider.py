import pandas
from os.path import join


def stitch_flat_to_pubchem(cid):
    assert cid.startswith("CID")
    return int(cid[3:]) - 1e8


def stitch_stereo_to_pubchem(cid):
    assert cid.startswith("CID")
    return int(cid[3:])


def main(root_path: str):
    # Read DrugBank terms
    url = "https://raw.githubusercontent.com/dhimmel/drugbank/3e87872db5fca5ac427ce27464ab945c0ceb4ec6/data/drugbank.tsv"
    drugbank_df = pandas.read_table(url)[["drugbank_id", "name"]].rename(
        columns={"name": "drugbank_name"}
    )

    # Pubchem to DrugBank mapping
    url = "https://raw.githubusercontent.com/dhimmel/drugbank/3e87872db5fca5ac427ce27464ab945c0ceb4ec6/data/mapping/pubchem.tsv"
    drugbank_map_df = pandas.read_table(url)

    columns = [
        "stitch_id_flat",
        "stitch_id_sterio",
        "umls_cui_from_label",
        "placebo",
        "frequency",
        "lower",
        "upper",
        "meddra_type",
        "umls_cui_from_meddra",
        "side_effect_name",
    ]
    freq_df = pandas.read_table(join(root_path, "meddra_freq.tsv.gz"), names=columns)
    freq_df.to_csv(join(root_path, "meddra_freq.csv"), index=False)

    columns = [
        "stitch_id_flat",
        "stitch_id_sterio",
        "umls_cui_from_label",
        "meddra_type",
        "umls_cui_from_meddra",
        "side_effect_name",
    ]
    se_df = pandas.read_table(join(root_path, "meddra_all_se.tsv.gz"), names=columns)
    se_df["pubchem_id"] = se_df.stitch_id_sterio.map(stitch_stereo_to_pubchem)
    se_df = drugbank_map_df.merge(se_df)
    se_df.to_csv(join(root_path, "meddra_all_se.csv"), index=False)

    se_df = se_df[["drugbank_id", "umls_cui_from_meddra", "side_effect_name"]]
    se_df = se_df.dropna()
    se_df = se_df.drop_duplicates(["drugbank_id", "umls_cui_from_meddra"])
    se_df = drugbank_df.merge(se_df)
    se_df = se_df.sort_values(["drugbank_name", "side_effect_name"])
    se_df.to_csv(join(root_path, "meddra_all_se.csv"))

    # Create a reference of side effect IDs and Names
    se_terms_df = se_df[["umls_cui_from_meddra", "side_effect_name"]].drop_duplicates()
    assert se_terms_df.side_effect_name.duplicated().sum() == 0
    se_terms_df = se_terms_df.sort_values("side_effect_name")
    se_terms_df.to_csv(join(root_path, "side-effect-terms.csv"), index=False)

    # Save side effects
    se_df.to_csv(join(root_path, "side-effects.csv"), index=False)

    columns = [
        "stitch_id_flat",
        "umls_cui_from_label",
        "method",
        "concept_name",
        "meddra_type",
        "umls_cui_from_meddra",
        "meddra_name",
    ]
    indication_df = pandas.read_table(
        join(root_path, "meddra_all_indications.tsv.gz"), names=columns
    )
    indication_df["pubchem_id"] = indication_df.stitch_id_flat.map(
        stitch_flat_to_pubchem
    )

    indication_df = drugbank_df.merge(drugbank_map_df.merge(indication_df))
    indication_df = indication_df.query("meddra_type == 'PT'")

    # Save indications
    indication_df.to_csv(join(root_path, "indications.csv"), index=False)


if __name__ == "__main__":
    root_path = "./data/databases/SIDER"
    main(root_path)

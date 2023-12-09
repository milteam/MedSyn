import click
import pandas as pd


@click.command()
@click.option("--wikimed-filepath", required=True)
@click.option("--rumedtop3-filepath", required=True)
@click.option("--output-filepath", required=True)
def main(wikimed_filepath: str, rumedtop3_filepath: str, output_filepath: str):
    rumed_df = pd.read_json(rumedtop3_filepath, lines=True)
    rumed_codes = set(rumed_df["code"].unique())

    wikimed = pd.read_csv(wikimed_filepath)
    wikimed.rename(
        columns={
            "МКБ-10": "ICD10_CODE",
            "Клинические проявления": "text"
        }, inplace=True)

    wikimed["ICD10_CAT"] = wikimed["ICD10_CODE"].str[:3]
    wikimed["ICD10_CODE"] = wikimed["ICD10_CODE"].str.strip("*+")
    columns = ["ICD10_CAT", "ICD10_CODE", "text"]
    wikimed = wikimed.loc[(~wikimed["text"].isna()) & (wikimed["ICD10_CAT"].isin(rumed_codes)), columns]

    wikimed.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    main()
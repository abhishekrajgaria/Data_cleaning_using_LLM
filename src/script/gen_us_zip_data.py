import pandas as pd
from constants import *


def generate_files(df, result_folderpath):

    unique_states = df["state"].unique()
    state_dfs = {value: df[df["state"] == value] for value in unique_states}
    for state, state_df in state_dfs.items():
        paragraph = ""
        for _, row in state_df.iterrows():
            city = row["city"]
            state = row["state"]
            zip_code = row["zip"]
            state_code = row["state_code"]
            paragraph += f"The city of {city}, located in the state of {state} (state code - {state_code}), is identified by the zip code {zip_code}.\n"

        with open(f"{result_folderpath}/{state}_zips.txt", "w") as file:
            file.write(paragraph)


def gen_us_zip_data():
    us_df = pd.read_csv(us_zip_data, sep="\t", header=None)
    us_df = us_df.iloc[:, 1:5]
    us_df.columns = ["zip", "city", "state", "state_code"]
    us_df.dropna(inplace=True)
    print(us_df.shape)
    print(us_df["state"].unique())

    generate_files(us_df, "../data/files/us_zip_data")


if __name__ == "__main__":
    gen_us_zip_data()

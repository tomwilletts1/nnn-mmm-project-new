import pandas as pd
import os


def load_weekly_data(path="./data/processed/"):
    dfs = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

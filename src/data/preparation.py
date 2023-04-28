import json
import numpy as np
import pandas as pd

from params import DATA_PATH


def prepare_folds(k):
    from sklearn.model_selection import StratifiedGroupKFold

    K = k

    df = pd.read_csv(DATA_PATH + "train.csv")

    sgkf = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=42)
    splits = sgkf.split(df, y=df["sign"], groups=df["participant_id"])

    df["fold"] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i

    df_folds = df[["participant_id", "sequence_id", "fold"]]
    df_folds.to_csv(f"../input/folds_{K}.csv", index=False)


def prepare_data(data_path="../input/", processed_folder=""):
    df = pd.read_csv(data_path + "train.csv")
    classes = json.load(open(data_path + "sign_to_prediction_index_map.json", "r"))

    df["target"] = df["sign"].map(classes)
    df["path"] = data_path + df["path"]

    df["processed_path"] = (
        DATA_PATH
        + processed_folder
        + df["participant_id"].astype(str)
        + "_"
        + df["sequence_id"].astype(str)
        + ".npy"
    )
    
    df['len'] = np.load('../output/raw_lens.npy')

    return df


def prepare_wsasl(data_path="../input/", processed_folder=""):
    df_wsasl = pd.read_csv(data_path + "df_wsasl.csv")
    classes = json.load(open(data_path + "sign_to_prediction_index_map.json", "r"))

    df_wsasl["target"] = df_wsasl["sign"].map(classes)
    df_wsasl = df_wsasl.dropna(axis=0)

    df_wsasl["processed_path"] = (
        data_path + processed_folder + df_wsasl["processed_path"]
    )

    return df_wsasl

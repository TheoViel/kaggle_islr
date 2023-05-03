import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from params import DATA_PATH


def prepare_folds(k):
    """
    Prepares K-fold cross-validation splits grouped by participant ID and stratified on sign.

    Args:
        k (int): The number of folds for cross-validation.
    """
    df = pd.read_csv(DATA_PATH + "train.csv")

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    splits = sgkf.split(df, y=df["sign"], groups=df["participant_id"])

    df["fold"] = -1
    for i, (_, val_idx) in enumerate(splits):
        df.loc[val_idx, "fold"] = i

    df_folds = df[["participant_id", "sequence_id", "fold"]]
    df_folds.to_csv(f"../input/folds_{k}.csv", index=False)


def prepare_data(data_path="../input/", processed_folder=""):
    """
    Loads and preprocesses the training data.

    Args:
        data_path (str): Path to the data folder. Defaults to "../input/".
        processed_folder (str): Folder containing the processed data. Defaults to "".

    Returns:
        pd.DataFrame: The preprocessed training data.
    """
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

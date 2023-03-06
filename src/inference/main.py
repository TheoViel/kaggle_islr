import os
import json
import glob
import numpy as np
import pandas as pd

from data.dataset import BreastDataset
from data.transforms import get_transfos
from model_zoo.models import define_model
from utils.logger import Config
from utils.metrics import tweak_thresholds
from utils.torch import load_model_weights
from inference.predict import predict, predict_tta


def kfold_inference_val(
    df,
    exp_folder,
    debug=False,
    save=True,
    use_tta=False,
    use_fp16=False,
    extract_fts=False
):
    """
    Main inference function for validation data.

    Args:
        df (pd DataFrame): Dataframe.
        exp_folder (str): Experiment folder.
        debug (bool, optional): Whether to use debug mode. Defaults to False.
        save (bool, optional): Whether to save predictions. Defaults to True.
        use_tta (bool, optional): Whether to use TTA. Defaults to False.
        use_fp16 (bool, optional): Whether to use fp16. Defaults to False.
        extract_fts (bool, optional): Whether to extract features. Defaults to False.

    Returns:
        np array [n x n_classes]: Predictions.
        np array [n x n_classes_aux]: Aux predictions.
    """
    predict_fct = predict_tta if use_tta else predict
    config = Config(json.load(open(exp_folder + "config.json", "r")))

    if "fold" not in df.columns:
        folds = pd.read_csv(config.folds_file)
        df = df.merge(folds, how="left", on=["patient_id", "image_id"])

    model = define_model(
        config.name,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_channels=config.n_channels,
        reduce_stride=config.reduce_stride,
        use_gem=config.use_gem,
        pretrained=False,
        replace_pad_conv=True,
    )
    model = model.cuda().eval()

    weights = [f for f in sorted(glob.glob(exp_folder + "*.pt")) if "fullfit" not in f]
    folds = [int(w[-4]) for w in weights]

    save_folder = ""
    if save:
        save_folder = exp_folder + "npy/"
        os.makedirs(save_folder, exist_ok=True)

    pred_oof = np.zeros((len(df), config.num_classes))
    pred_oof_aux = np.zeros((len(df), config.num_classes_aux))
    for fold in folds:
        print(f"\n- Fold {fold + 1}")

        val_idx = list(df[df["fold"] == fold].index)
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)

        if debug:
            df_val = df_val.head(4)

        dataset = BreastDataset(
            df_val,
            transforms=get_transfos(augment=False, resize=config.resize),
        )

        weight = weights[folds.index(fold)]
        model = load_model_weights(model, weight, verbose=1)

        pred_val, pred_val_aux, fts = predict_fct(
            model,
            dataset,
            config.loss_config,
            batch_size=config.data_config["val_bs"],
            use_fp16=use_fp16,
        )

        df_val["pred"] = pred_val
        dfg = (
            df_val[["patient_id", "laterality", "pred", "cancer"]]
            .groupby(["patient_id", "laterality"])
            .mean()
        )

        th, _, pf1 = tweak_thresholds(dfg["cancer"].values, dfg["pred"].values)
        print(f" -> Scored {pf1 :.4f}      (th={th:.2f})")

        if debug:
            return pred_val, pred_val_aux

        if save:
            np.save(exp_folder + f"pred_val_inf_{fold}.npy", pred_val)
            if config.num_classes_aux:
                np.save(exp_folder + f"pred_val_aux_inf_{fold}.npy", pred_val_aux)
            if extract_fts:
                np.save(exp_folder + f"fts_{fold}.npy", fts)

        pred_oof[val_idx] = pred_val
        if config.num_classes_aux:
            pred_oof_aux[val_idx] = pred_val_aux

    if save:
        np.save(exp_folder + "pred_oof_inf.npy", pred_oof)
        if config.num_classes_aux:
            np.save(exp_folder + "pred_oof_aux_inf.npy", pred_oof_aux)

    return pred_oof, pred_oof_aux

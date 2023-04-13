import os
import json
import glob
import torch
import numpy as np
import pandas as pd

from data.dataset import SignDataset
from model_zoo.models import define_model
from utils.logger import Config
from utils.torch import load_model_weights
from utils.metrics import accuracy
from inference.predict import predict, predict_tta


def uniform_soup(model, weights, device="cpu", by_name=False):
    if not isinstance(weights, list):
        weights = [weights]

    model = model.to(device)
    model_dict = model.state_dict()

    soups = {key:[] for key in model_dict}
    
    for i, model_path in enumerate(weights):
        weight = torch.load(model_path, map_location=device)
        weight_dict = weight.state_dict() if hasattr(weight, "state_dict") else weight
        
        if by_name:
            weight_dict = {k: v for k, v in weight_dict.items() if k in model_dict}

        for k, v in weight_dict.items():
            soups[k].append(v)

    if 0 < len(soups):
        soups = {
            k: (torch.sum(torch.stack(v), axis = 0) / len(v)).type(v[0].dtype)
            for k, v in soups.items() if len(v) != 0
        }
        model_dict.update(soups)
        model.load_state_dict(model_dict)

    return model


def kfold_inference_val(
    df,
    exp_folder,
    debug=False,
    save=True,
    use_tta=False,
    use_fp16=False,
    train=False
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
        embed_dim=config.embed_dim,
        transfo_dim=config.transfo_dim,
        dense_dim=config.dense_dim,
        transfo_heads=config.transfo_heads,
        transfo_layers=config.transfo_layers,
        drop_rate=config.drop_rate,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_landmarks=config.n_landmarks,
        max_len=config.max_len,
        verbose=(config.local_rank == 0),
    )
    model = model.cuda().eval()

    pred_oof = np.zeros((config.k, len(df), config.num_classes))
    for fold in config.selected_folds:
        print(f"\n- Fold {fold + 1}")
        
        if config.model_soup:
            weights = [f for f in sorted(glob.glob(exp_folder + f"*_{fold}_*.pt"))][-1:]
#             weights += [f for f in sorted(glob.glob("../logs/2023-03-30/3/" + f"*_{fold}.pt")) if "fullfit" not in f]
            print("Soup :", weights)
            model = uniform_soup(model, weights)
            model = model.cuda().eval()
        else:
            weights = [f for f in sorted(glob.glob(exp_folder + f"*_{fold}.pt"))][0]
            model = load_model_weights(model, weights, verbose=1)

        if train:
            val_idx = list(df[df["fold"] != fold].index)
        else:
            val_idx = list(df[df["fold"] == fold].index)
        df_val = df.iloc[val_idx].copy().reset_index(drop=True)

        if debug:
            df_val = df_val.head(4)

        dataset = SignDataset(
            df_val,
            max_len=config.max_len,
            resize_mode=config.resize_mode,
            train=False,
        )

        pred_val, pred_val_aux = predict_fct(
            model,
            dataset,
            config.loss_config,
            batch_size=config.data_config["val_bs"],
            use_fp16=use_fp16,
        )

        acc = accuracy(df_val["target"].values, pred_val)
        print(f"\n -> Accuracy : {acc:.4f}")

        if debug:
            return pred_val, pred_val_aux

        if save:
            np.save(exp_folder + f"pred_val_inf_{fold}.npy", pred_val)
            
        pred_oof[fold, val_idx] = pred_val

#         break

    if not train:
        pred_oof = pred_oof.sum(0)

    acc = accuracy(df["target"].values, pred_oof)
    print(f"\n\n -> CV Accuracy : {acc:.4f}")
    if save:
        np.save(exp_folder + "pred_oof_inf.npy", pred_oof)

    return pred_oof

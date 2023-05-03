import json
import torch
import numpy as np
import pandas as pd

from data.dataset import SignDataset
from model_zoo.models import define_model
from utils.logger import Config
from utils.torch import load_model_weights
from utils.metrics import accuracy
from inference.predict import predict


def uniform_soup(model, weights, device="cpu", by_name=False, weighting="uniform"):
    """
    Creates a "soup" model by combining the weights from multiple models.

    Args:
        model (torch.nn.Module): The base model.
        weights (list or torch.nn.Module): List of model weights or individual model weight.
        device (str, optional): Device to load the model on. Defaults to "cpu".
        by_name (bool, optional): Whether to match weights by name. Defaults to False.
        weighting (str, optional): Weighting to apply ("uniform"/"linear"). Defaults to "uniform".

    Returns:
        torch.nn.Module: The combined "soup" model.
    """
    if not isinstance(weights, list):
        weights = [weights]

    model = model.to(device)
    model_dict = model.state_dict()

    soups = {key: [] for key in model_dict}

    for i, model_path in enumerate(weights):
        weight = torch.load(model_path, map_location=device)
        weight_dict = weight.state_dict() if hasattr(weight, "state_dict") else weight

        if by_name:
            weight_dict = {k: v for k, v in weight_dict.items() if k in model_dict}

        for k, v in weight_dict.items():
            soups[k].append(v)

    if weighting == "uniform":
        w = torch.ones(len(weights))
    else:
        w = torch.from_numpy(np.arange(1, len(weights) + 1))
        print("Weighting", w.numpy())

    if 0 < len(soups):
        soups = {
            k: (
                torch.sum(
                    torch.stack(v) * w.view([-1] + [1] * len(v[0].size())), axis=0
                )
                / w.sum()
            ).type(v[0].dtype)
            for k, v in soups.items()
            if len(v) != 0
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
    train=False,
    use_mt=False,
    distilled=False,
    n_soup=0
):
    """
    Perform k-fold cross-validation for model inference on the validation set.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        exp_folder (str): Path to the experiment folder.
        debug (bool, optional): Whether to run in debug mode. Defaults to False.
        save (bool, optional): Whether to save the predictions. Defaults to True.
        use_tta (bool, optional): Whether to use test time augmentation. Defaults to False.
        use_fp16 (bool, optional): Whether to use mixed precision inference. Defaults to False.
        train (bool, optional): Whether to perform inference on the training set. Defaults to False.
        use_mt (bool, optional): Whether to use model teacher. Defaults to False.
        distilled (bool, optional): Whether to use distilled model. Defaults to False.
        n_soup (int, optional): Number of models to use for model soup. Defaults to 0.

    Returns:
        np.ndarray: Array containing the predicted probabilities for each class.
    """
    assert not use_tta, "TTA not implemented"
    predict_fct = predict  # predict_tta if use_tta else predict

    config = Config(json.load(open(exp_folder + "config.json", "r")))

    if "fold" not in df.columns:
        folds = pd.read_csv(config.folds_file)
        df = df.merge(folds, how="left", on=["patient_id", "image_id"])

    if distilled:
        try:
            distill_transfo_dim = config.mt_config['distill_transfo_dim']
            distill_dense_dim = config.mt_config['distill_dense_dim']
            distill_transfo_layers = config.mt_config['distill_transfo_layers']
        except KeyError:
            distill_transfo_dim = 576  # 512 / 768
            distill_dense_dim = 192  # 256
            distill_transfo_layers = 3

    model = define_model(
        config.name,
        embed_dim=config.embed_dim,
        transfo_dim=config.transfo_dim if not distilled else distill_transfo_dim,
        dense_dim=config.dense_dim if not distilled else distill_dense_dim,
        transfo_heads=config.transfo_heads,
        transfo_layers=config.transfo_layers if not distilled else distill_transfo_layers,
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

        if config.model_soup and n_soup > 1:
            if use_mt:
                weights = [
                    exp_folder + f"{config.name}_teacher_{fold}_{ep}.pt"
                    for ep in range(config.epochs - n_soup, config.epochs + 1)
                ]
            elif distilled:
                weights = [
                    exp_folder + f"{config.name}_distilled_{fold}_{ep}.pt"
                    for ep in range(config.epochs - n_soup, config.epochs + 1)
                ]
            else:
                weights = [
                    exp_folder + f"{config.name}_{fold}_{ep}.pt"
                    for ep in range(config.epochs - n_soup, config.epochs + 1)
                ]

            weights = weights[-n_soup:]
            print("\nSoup :", weights)
            model = uniform_soup(model, weights)
            model = model.cuda().eval()

        else:
            if use_mt:
                weights = exp_folder + f"{config.name}_teacher_{fold}.pt"
            elif distilled:
                weights = exp_folder + f"{config.name}_distilled_{fold}.pt"
            else:
                weights = exp_folder + f"{config.name}_{fold}.pt"

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
        if distilled:
            name = "pred_oof_dist"
        elif use_mt:
            name = "pred_oof_mt"
        else:
            name = "pred_oof_inf"
        if n_soup > 1:
            name += "_soup"
        np.save(exp_folder + f"{name}.npy", pred_oof)

    return pred_oof

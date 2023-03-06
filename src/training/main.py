import gc
import re
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel

from params import DATA_PATH
from training.train import fit
from model_zoo.models import define_model

from data.dataset import BreastDataset
from data.transforms import get_transfos
from data.preparation import (
    prepare_cbis_data,
    prepare_cmmd_data,
    prepare_pasm_data,
    prepare_pl_data
)

from utils.torch import seed_everything, count_parameters, save_model_weights
from utils.metrics import tweak_thresholds


def train(config, df_train, df_val, fold, log_folder=None, run=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
        run (None or Nepture run): Nepture run. Defaults to None.

    Returns:
        np array [len(df_train) x num_classes]: Validation predictions.
    """
    crop = "crops" in config.img_folder

    dataset_cls = BreastCropDataset if crop else BreastDataset

    train_dataset = dataset_cls(
        df_train,
        transforms=get_transfos(resize=config.resize, strength=config.aug_strength),
        sampler_weights=config.data_config["sampler_weights"],
    )

    val_dataset = dataset_cls(
        df_val,
        transforms=get_transfos(augment=False, resize=config.resize),
    )

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(
            ".pt"
        ) or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[0]
    else:
        pretrained_weights = None

    model = define_model(
        config.name,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_channels=config.n_channels,
        pretrained_weights=pretrained_weights,
        drop_rate=config.drop_rate,
        drop_path_rate=config.drop_path_rate,
        use_gem=config.use_gem,
        reduce_stride=config.reduce_stride,
        verbose=(config.local_rank == 0),
        crop=crop,
    ).cuda()

    if config.distributed:
        if config.syncbn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DistributedDataParallel(
            model,
            device_ids=[config.local_rank],
            find_unused_parameters=False,
            broadcast_buffers=config.syncbn,
        )

    try:
        model = torch.compile(model, mode="reduce-overhead")
        if config.local_rank == 0:
            print("Using torch 2.0 acceleration !\n")
    except Exception:
        pass

    model.zero_grad(set_to_none=True)
    model.train()

    n_parameters = count_parameters(model)

    if config.local_rank == 0:
        print(f"    -> {len(train_dataset)} training images")
        print(f"    -> {len(val_dataset)} validation images")
        print(f"    -> {n_parameters} trainable parameters\n")

    pred_val = fit(
        model,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        epochs=config.epochs,
        verbose_eval=config.verbose_eval,
        device=config.device,
        use_fp16=config.use_fp16,
        distributed=config.distributed,
        local_rank=config.local_rank,
        world_size=config.world_size,
        run=run,
        fold=fold,
    )

    if (log_folder is not None) and (config.local_rank == 0):
        save_model_weights(
            model.module if config.distributed else model,
            f"{config.name.split('/')[-1]}_{fold}.pt",
            cp_folder=log_folder,
        )

    del (model, train_dataset, val_dataset)
    torch.cuda.empty_cache()
    gc.collect()

    return pred_val


def k_fold(config, df, df_extra=None, log_folder=None, run=None):
    """_summary_

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        df_extra (pandas dataframe or None, optional): Extra metadata. Defaults to None.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
        run (None or Nepture run): Nepture run. Defaults to None.

    Returns:
        np array [len(df) x num_classes]: Oof predictions.
    """
    if "fold" not in df.columns:
        folds = pd.read_csv(config.folds_file)
        df = df.merge(folds, how="left", on=["patient_id", "image_id"])

    pred_oof = np.zeros((len(df), config.num_classes))
    for fold in range(config.k):
        if fold in config.selected_folds:
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fold {fold + 1} / {config.k}  -------------\n"
                )

            seed_everything(
                int(re.sub(r"\W", "", config.name), base=36) % 2**31 + fold
            )
            train_idx = list(df[df["fold"] != fold].index)
            val_idx = list(df[df["fold"] == fold].index)

            df_train = df.iloc[train_idx].copy().reset_index(drop=True)

            if config.use_cbis:
                df_cbis = prepare_cbis_data(DATA_PATH, "cbis_" + config.img_folder)
                df_train = pd.concat([df_train, df_cbis], ignore_index=True)

            if config.use_cmmd:
                df_cbis = prepare_cmmd_data(DATA_PATH, "cmmd_" + config.img_folder)
                df_train = pd.concat([df_train, df_cbis], ignore_index=True)

            if config.use_pasm:
                df_pasm = prepare_pasm_data(DATA_PATH, "pasm_" + config.img_folder)
                df_train = pd.concat([df_train, df_pasm], ignore_index=True)

            if config.use_pl:
                df_pl = prepare_pl_data(DATA_PATH, "vindr_" + config.img_folder, fold=fold)
                df_train = pd.concat([df_train, df_pl], ignore_index=True)

            df_val = df.iloc[val_idx].copy().reset_index(drop=True)

            pred_val = train(
                config, df_train, df_val, fold, log_folder=log_folder, run=run
            )

            if log_folder is None:
                return pred_val

            if config.local_rank == 0:
                np.save(log_folder + f"pred_val_{fold}", pred_val)
                df_val.to_csv(log_folder + f"df_val_{fold}.csv", index=False)
                pred_oof[val_idx] = pred_val

                if run is not None:
                    run[f"fold_{fold}/pred_val"].upload(
                        log_folder + f"df_val_{fold}.csv"
                    )

    if config.selected_folds == list(range(config.k)) and (config.local_rank == 0):
        if config.num_classes == 1:
            df["pred"] = pred_oof.flatten()
        elif config.num_classes == 2:
            df["pred"] = pred_oof[:, 1].flatten()

        dfg = (
            df[["patient_id", "laterality", "pred", "cancer"]]
            .groupby(["patient_id", "laterality"])
            .mean()
        )
        _, _, pf1 = tweak_thresholds(dfg["cancer"].values, dfg["pred"].values)
        auc = roc_auc_score(dfg["cancer"].values, dfg["pred"].values)

        print(f"\n\n -> CV pF1 : {pf1:.4f}")

        if log_folder is not None:
            folds.to_csv(log_folder + "folds.csv", index=False)
            np.save(log_folder + "pred_oof.npy", pred_oof)
            df.to_csv(log_folder + "df.csv", index=False)

            if run is not None:
                run["global/logs"].upload(log_folder + "logs.txt")
                run["global/pred_oof"].upload(log_folder + "pred_oof.npy")
                run["global/cv"] = pf1
                run["global/auc"] = auc

    if config.fullfit:
        for ff in range(config.n_fullfit):
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                )
            seed_everything(config.seed + ff)

            df_train = df.copy()

            if config.use_cbis:
                df_cbis = prepare_cbis_data(DATA_PATH, "cbis_" + config.img_folder)
                df_train = pd.concat([df_train, df_cbis], ignore_index=True)
            if config.use_cmmd:
                df_cbis = prepare_cmmd_data(DATA_PATH, "cmmd_" + config.img_folder)
                df_train = pd.concat([df_train, df_cbis], ignore_index=True)
            if config.use_pasm:
                df_pasm = prepare_pasm_data(DATA_PATH, "pasm_" + config.img_folder)
                df_train = pd.concat([df_train, df_pasm], ignore_index=True)
            if config.use_pl:
                df_pl = prepare_pl_data(DATA_PATH, "vindr_" + config.img_folder, fold="fullfit")
                df_train = pd.concat([df_train, df_pl], ignore_index=True)

            train(
                config,
                df_train,
                df.tail(100).reset_index(drop=True),
                f"fullfit_{ff}",
                log_folder=log_folder,
                run=run,
            )

    if run is not None:
        print()
        run.stop()

    return pred_oof

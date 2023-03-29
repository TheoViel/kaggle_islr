import gc
import re
import glob
import torch
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

from params import DATA_PATH
from training.train import fit
from model_zoo.models import define_model

from data.dataset import SignDataset

from utils.torch import seed_everything, count_parameters, save_model_weights
from utils.metrics import accuracy


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
    train_dataset = SignDataset(
        df_train,
        max_len=config.max_len,
        aug_strength=config.aug_strength,
        resize_mode=config.resize_mode,
        train=True,
    )

    val_dataset = SignDataset(
        df_val,
        max_len=config.max_len,
        resize_mode=config.resize_mode,
        train=False,
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
        pretrained_weights=pretrained_weights,
        embed_dim=config.embed_dim,
        transfo_dim=config.transfo_dim,
        transfo_heads=config.transfo_heads,
        transfo_layers=config.transfo_layers,
        drop_rate=config.drop_rate,
        num_classes=config.num_classes,
        num_classes_aux=config.num_classes_aux,
        n_landmarks=config.n_landmarks,
        verbose=(config.local_rank == 0),
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
    """
    Trains a k-fold.

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
        df = df.merge(folds, how="left", on=["participant_id", "sequence_id"])

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
            df_val = df.iloc[val_idx].copy().reset_index(drop=True)
            
            if df_extra is not None:
                df_train = pd.concat([df_train, df_extra], ignore_index=True)

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
        acc = accuracy(df["target"].values, pred_oof)
        print(f"\n\n -> CV Accuracy : {acc:.4f}")

        if log_folder is not None:
            folds.to_csv(log_folder + "folds.csv", index=False)
            np.save(log_folder + "pred_oof.npy", pred_oof)
            df.to_csv(log_folder + "df.csv", index=False)

            if run is not None:
                run["global/logs"].upload(log_folder + "logs.txt")
                run["global/pred_oof"].upload(log_folder + "pred_oof.npy")
                run["global/cv"] = acc

    if config.fullfit:
        for ff in range(config.n_fullfit):
            if config.local_rank == 0:
                print(
                    f"\n-------------   Fullfit {ff + 1} / {config.n_fullfit} -------------\n"
                )
            seed_everything(config.seed + ff)

            df_train = df.copy()

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

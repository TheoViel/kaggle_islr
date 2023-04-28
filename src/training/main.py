import gc
import re
import glob
import torch
import numpy as np
import pandas as pd
from torch.nn.parallel import DistributedDataParallel

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
    if config.local_rank == 0:
        print("    -> Data Preparation")
    train_dataset = SignDataset(
        df_train,
        max_len=config.max_len,
        aug_strength=config.aug_strength,
        resize_mode=config.resize_mode,
        train=True,
        dist=config.mt_config['distill'],
    )

    val_dataset = SignDataset(
        df_val,
        max_len=config.max_len,
        resize_mode=config.resize_mode,
        train=False,
    )

    if config.epochs > 10:
        train_dataset.fill_buffer()
        val_dataset.fill_buffer()

    if config.pretrained_weights is not None:
        if config.pretrained_weights.endswith(
            ".pt"
        ) or config.pretrained_weights.endswith(".bin"):
            pretrained_weights = config.pretrained_weights
        else:  # folder
            pretrained_weights = glob.glob(config.pretrained_weights + f"*_{fold}.pt")[
                0
            ]
    else:
        pretrained_weights = None

    model = define_model(
        config.name,
        pretrained_weights=pretrained_weights,
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
    ).cuda()

    model_teacher = define_model(
        config.name,
        pretrained_weights=pretrained_weights,
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
    ).cuda()

    model_distilled = None
    if config.mt_config['distill']:
#         if config.transfo_dim == 1024:
#             distill_transfo_dim = 576  # 768
#             distill_dense_dim = 192
#             distill_transfo_layers = 3
#         elif config.transfo_dim == 768:
#             distill_transfo_dim = 576  # 512
#             distill_dense_dim = 192
#             distill_transfo_layers = 3  # 2
#         else:
        distill_transfo_dim = 576  # 768
        distill_dense_dim = 192
        distill_transfo_layers = 3
#             raise NotImplementedError

        model_distilled = define_model(
            config.name,
            pretrained_weights=pretrained_weights,
            embed_dim=config.embed_dim,
            transfo_dim=config.mt_config['distill_transfo_dim'],
            dense_dim=config.mt_config['distill_dense_dim'],
            transfo_heads=config.transfo_heads,
            transfo_layers=config.mt_config['distill_transfo_layers'],
            drop_rate=config.drop_rate,
            num_classes=config.num_classes,
            num_classes_aux=config.num_classes_aux,
            n_landmarks=config.n_landmarks,
            max_len=config.max_len,
            verbose=0,
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
        
        if model_distilled is not None:
            model_distilled = DistributedDataParallel(
                model_distilled,
                device_ids=[config.local_rank],
                find_unused_parameters=False,
                broadcast_buffers=config.syncbn,
            )

#     try:
#         model = torch.compile(model, mode="reduce-overhead")
#         if config.local_rank == 0:
#             print("Using torch 2.0 acceleration !\n")
#     except Exception:
#         pass

    model.zero_grad(set_to_none=True)
    model.train()
    
    if model_distilled is not None:
        model_distilled.zero_grad(set_to_none=True)
        model_distilled.train()

    for param in model_teacher.parameters():
        param.detach_()
        
    n_parameters = count_parameters(model)
    if config.local_rank == 0:
        print(f"    -> {len(train_dataset)} training images")
        print(f"    -> {len(val_dataset)} validation images")
        print(f"    -> {n_parameters} trainable parameters")
        if model_distilled is not None:
            dist_parameters = count_parameters(model_distilled)
            print(f"    -> {dist_parameters} distilled parameters\n")
        else:
            print("")

    pred_val = fit(
        model,
        model_teacher,
        model_distilled,
        train_dataset,
        val_dataset,
        config.data_config,
        config.loss_config,
        config.optimizer_config,
        config.mt_config,
        epochs=config.epochs,
        verbose_eval=config.verbose_eval,
        model_soup=config.model_soup,
        use_fp16=config.use_fp16,
        distributed=config.distributed,
        local_rank=config.local_rank,
        world_size=config.world_size,
        log_folder=log_folder,
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

    # Train oof for confidence
    #     oof = np.load("../logs/2023-04-11/27/pred_oof_train.npy")
    oof = np.stack(
        [np.load("../logs/2023-04-09/2/pred_oof.npy") for _ in range(config.k)], 0
    )  # LEAKY ?
    oof_ = [oof[:, i, j] for i, j in enumerate(df["target"].values)]
    oof_ = np.stack(oof_).T
    for i in range(len(oof_)):
        df[f"pred_{i}"] = oof_[i]
        
#     df['lens'] = np.load('../output/lens.npy')

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

            df_train = df.iloc[train_idx].reset_index(drop=True)
            df_val = df.iloc[val_idx].reset_index(drop=True)
            
#             df_train = df_train[df_train['lens'] <= 5].reset_index(drop=True)

            if df_extra is not None:
                df_train = pd.concat([df_train, df_extra], ignore_index=True)

            # df_train = df_train[
            #     df_train[f'pred_{fold}'] > np.percentile(df_train[f'pred_{fold}'], 10)
            # ].reset_index(drop=True)

#             df_train = df_train[df_train['participant_id'] != 29302].reset_index(drop=True)
            # two_hands = np.concatenate([
            #     np.load('../output/two_hands_others.npy'),
            #     np.load('../output/two_hands_29302.npy')
            # ])
            # df_train = df_train[~df_train['sequence_id'].isin(two_hands)].reset_index(drop=True)

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

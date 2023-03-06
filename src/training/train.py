import gc
import time
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import get_linear_schedule_with_warmup

from data.loader import define_loaders
from training.losses import BreastLoss
from training.optim import define_optimizer

from training.mix import cutmix_data, Mixup
from utils.metrics import tweak_thresholds
from utils.torch import sync_across_gpus


def evaluate(
    model,
    val_loader,
    loss_config,
    loss_fct,
    use_fp16=False,
    distributed=False,
    world_size=0,
    local_rank=0,
):
    """
    Evaluates a model.

    Args:
        model (torch model): Model.
        val_loader (DataLoader): Data Loader.
        loss_config (dict): Loss config.
        loss_fct (nn.Module): Loss function.
        use_fp16 (bool, optional): Whether to use fp16. Defaults to False.
        distributed (bool, optional): Whether training is distributed. Defaults to False.
        world_size (int, optional): World size. Defaults to 0.
        local_rank (int, optional): Local rank. Defaults to 0.

    Returns:
        np array [n x num_classes]: Predictions.
        np array [n x num_classes_aux]: Aux predictions.
        torch tensor: Val loss.
    """
    model.eval()
    val_losses = []
    preds, preds_aux = [], []

    with torch.no_grad():
        for x, y, y_aux in val_loader:
            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, y_pred_aux = model(x.cuda())
                loss = loss_fct(
                    y_pred.detach(), y_pred_aux.detach(), y.cuda(), y_aux.cuda()
                )

            val_losses.append(loss.detach())

            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)

            preds.append(y_pred.detach())
            preds_aux.append(y_pred_aux.detach())

    val_losses = torch.stack(val_losses)
    preds = torch.cat(preds, 0)
    preds_aux = torch.cat(preds_aux, 0)

    if distributed:
        val_losses = sync_across_gpus(val_losses, world_size)
        preds = sync_across_gpus(preds, world_size)
        if model.module.num_classes_aux:
            preds_aux = sync_across_gpus(preds_aux, world_size)
        torch.distributed.barrier()

    if local_rank == 0:
        preds = preds.cpu().numpy()
        preds_aux = preds_aux.cpu().numpy()
        val_loss = val_losses.cpu().numpy().mean()
        return preds, preds_aux, val_loss
    else:
        return 0, 0, 0


def fit(
    model,
    train_dataset,
    val_dataset,
    data_config,
    loss_config,
    optimizer_config,
    epochs=1,
    verbose_eval=1,
    use_fp16=False,
    distributed=False,
    local_rank=0,
    world_size=1,
    run=None,
    fold=0,
):
    """
    Trains a model.

    Args:
        model (torch model): Model.
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        data_config (dict): Data config.
        loss_config (dict): Loss config.
        optimizer_config (dict): Optimizer config.
        epochs (int, optional): Number of epochs. Defaults to 1.
        verbose_eval (int, optional): Steps for evaluation. Defaults to 1.
        use_fp16 (bool, optional): Whether to use fp16. Defaults to False.
        distributed (bool, optional): Whether training is distributed. Defaults to False.
        world_size (int, optional): World size. Defaults to 0.
        local_rank (int, optional): Local rank. Defaults to 0.
        run (neptune run, optional): Neptune run. Defaults to None.
        fold (int, optional): Fold number. Defaults to 0.

    Returns:
        np array [n x num_classes]: Predictions.
    """
    scaler = torch.cuda.amp.GradScaler()

    optimizer = define_optimizer(
        model,
        optimizer_config["name"],
        lr=optimizer_config["lr"],
        betas=optimizer_config["betas"],
    )

    loss_fct = BreastLoss(loss_config)

    train_loader, val_loader = define_loaders(
        train_dataset,
        val_dataset,
        batch_size=data_config["batch_size"],
        val_bs=data_config["val_bs"],
        use_weighted_sampler=data_config["use_weighted_sampler"],
        use_balanced_sampler=data_config["use_balanced_sampler"],
        use_len_sampler=data_config["use_len_sampler"],
        use_custom_collate=data_config["use_custom_collate"],
        distributed=distributed,
        world_size=world_size,
        local_rank=local_rank,
    )

    if data_config["mix"] == "cutmix":
        mix = cutmix_data  # TODO : use class
    else:
        mix = Mixup(data_config["mix_alpha"], data_config["additive_mix"])

    # LR Scheduler
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(optimizer_config["warmup_prop"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    step = 1
    pf1, auc, acc = 0, 0, 0
    avg_losses = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        for x, y, y_aux in train_loader:
            if np.random.random() < data_config["mix_proba"]:
                x, y, y_aux = mix(x, y, y_aux)

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, y_pred_aux = model(x.cuda())
                loss = loss_fct(y_pred, y_pred_aux, y.cuda(), y_aux.cuda())

            scaler.scale(loss).backward()
            avg_losses.append(loss.detach())

            scaler.unscale_(optimizer)
            if optimizer_config["max_grad_norm"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    optimizer_config["max_grad_norm"],
                    #                 error_if_nonfinite=False,
                )
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()

            model.zero_grad(set_to_none=True)

            if distributed:
                torch.cuda.synchronize()

            if scale == scaler.get_scale():
                scheduler.step()
            step += 1

            if (step % verbose_eval) == 0 or step - 1 >= epochs * len(train_loader):
                if 0 <= epochs * len(train_loader) - step < verbose_eval:
                    continue

                avg_losses = torch.stack(avg_losses)
                if distributed:
                    avg_losses = sync_across_gpus(avg_losses, world_size)
                avg_loss = avg_losses.cpu().numpy().mean()

                preds, preds_aux, avg_val_loss = evaluate(
                    model,
                    val_loader,
                    loss_config,
                    loss_fct,
                    use_fp16=use_fp16,
                    distributed=distributed,
                    world_size=world_size,
                    local_rank=local_rank,
                )

                if local_rank == 0:
                    preds = preds[: len(val_dataset)]
                    preds_aux = preds_aux[: len(val_dataset)]
                    num_classes = preds.shape[-1] if len(preds.shape) > 1 else 1
                    try:
                        if num_classes == 1:
                            val_dataset.df["pred"] = preds.flatten()
                        elif num_classes == 2:
                            val_dataset.df["pred"] = preds[:, 1].flatten()
                        dfg = (
                            val_dataset.df[
                                ["patient_id", "laterality", "pred", "cancer"]
                            ]
                            .groupby(["patient_id", "laterality"])
                            .mean()
                        )
                        _, _, pf1 = tweak_thresholds(
                            dfg["cancer"].values, dfg["pred"].values
                        )
                        auc = roc_auc_score(dfg["cancer"].values, dfg["pred"].values)
                    except Exception:  # Pretraining
                        acc = accuracy_score(val_dataset.targets, preds.argmax(-1))

                    dt = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    step_ = step * world_size

                    s = f"Epoch {epoch:02d}/{epochs:02d} (step {step_:04d}) \t"
                    s = s + f"lr={lr:.1e} \t t={dt:.0f}s  \t loss={avg_loss:.3f}"
                    s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                    s = s + f"    pf1={pf1:.3f}" if pf1 else s
                    s = s + f"    auc={auc:.3f}" if auc else s
                    s = s + f"    acc={acc:.3f}" if acc else s

                    print(s)

                if run is not None:
                    run[f"fold_{fold}/train/epoch"].log(epoch, step=step_)
                    run[f"fold_{fold}/train/loss"].log(avg_loss, step=step_)
                    run[f"fold_{fold}/train/lr"].log(lr, step=step_)
                    run[f"fold_{fold}/val/loss"].log(avg_val_loss, step=step_)
                    run[f"fold_{fold}/val/pf1"].log(pf1, step=step_)
                    run[f"fold_{fold}/val/auc"].log(auc, step=step_)

                start_time = time.time()
                avg_losses = []
                model.train()

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    if distributed:
        torch.distributed.barrier()

    return preds

import gc
import time
import torch
# import numpy as np
# from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data.loader import define_loaders
from training.losses import SignLoss, ConsistencyLoss
from training.optim import define_optimizer, update_teacher_params, AWP
from model_zoo.utils import modify_drop
from utils.metrics import accuracy
from utils.torch import sync_across_gpus, save_model_weights


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
    
    try:
        num_classes_aux = model.module.num_classes_aux
    except AttributeError:
        num_classes_aux = model.num_classes_aux

    with torch.no_grad():
        for data, _ in val_loader:
            for k in data.keys():
                data[k] = data[k].cuda()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, y_pred_aux = model(data)
                
                if isinstance(y_pred_aux, list):
                    y_pred_aux = [y.detach() for y in y_pred_aux]
                else:
                    y_pred_aux = y_pred_aux.detach()

                loss = loss_fct(y_pred.detach(), y_pred_aux, data["target"], 0)

            val_losses.append(loss.detach())

            if loss_config["activation"] == "sigmoid":
                y_pred = y_pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                y_pred = y_pred.softmax(-1)

            preds.append(y_pred.detach())
            preds_aux.append(y_pred_aux)

    val_losses = torch.stack(val_losses)
    preds = torch.cat(preds, 0)
    if num_classes_aux:
        preds_aux = torch.cat(preds_aux, 0)
    else:
        preds_aux = 0

    if distributed:
        val_losses = sync_across_gpus(val_losses, world_size)
        preds = sync_across_gpus(preds, world_size)
        if num_classes_aux:
            preds_aux = sync_across_gpus(preds_aux, world_size)
        torch.distributed.barrier()

    if local_rank == 0:
        preds = preds.cpu().numpy()
        if num_classes_aux:
            preds_aux = preds_aux.cpu().numpy()
        val_loss = val_losses.cpu().numpy().mean()
        return preds, preds_aux, val_loss
    else:
        return 0, 0, 0


def fit(
    model,
    model_teacher,
    train_dataset,
    val_dataset,
    data_config,
    loss_config,
    optimizer_config,
    mt_config,
    epochs=1,
    verbose_eval=1,
    use_fp16=False,
    model_soup=False,
    distributed=False,
    local_rank=0,
    world_size=1,
    log_folder=None,
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
        weight_decay=optimizer_config["weight_decay"],
    )

    train_loader, val_loader = define_loaders(
        train_dataset,
        val_dataset,
        batch_size=data_config["batch_size"],
        val_bs=data_config["val_bs"],
        use_len_sampler=data_config["use_len_sampler"],
        distributed=distributed,
        world_size=world_size,
        local_rank=local_rank,
    )

    # LR Scheduler
    num_training_steps = epochs * len(train_loader)
    num_warmup_steps = int(optimizer_config["warmup_prop"] * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    loss_fct = SignLoss(loss_config)

    mt_config["rampup_length"] = int(num_training_steps * mt_config["rampup_prop"])
    consistency_loss = ConsistencyLoss(mt_config)
    
#     if optimizer_config['use_awp']:
#         awp = AWP(
#             model,
#             optimizer,
#             adv_lr=optimizer_config['awp_lr'],
#             adv_eps=optimizer_config['awp_eps'],
#         )

    step = 1
    acc = 0
    acc_teach = 0
    avg_losses = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):
#         if epoch == 1:
#             modify_drop(model, factor=2)
#         elif epoch == int(epochs * 0.25):
#             modify_drop(model, factor=0.5)
#         elif epoch == int(epochs * 0.5):
#             modify_drop(model, factor=0.5)
#         elif epoch == int(epochs * 0.75):
#             modify_drop(model, factor=0.5)

        if epoch in [min(50, epochs // 2)]:  # , 100, 110]:
            if epoch == min(50, epochs // 2):
                train_dataset.aug_strength = min(train_dataset.aug_strength, 2)
#             elif epoch == 100:
#                 train_dataset.aug_strength = min(train_dataset.aug_strength, 1)
#             elif epoch == 110:
#                 train_dataset.aug_strength = 0

            train_loader, val_loader = define_loaders(
                train_dataset,
                val_dataset,
                batch_size=data_config["batch_size"],
                val_bs=data_config["val_bs"],
                use_len_sampler=data_config["use_len_sampler"],
                distributed=distributed,
                world_size=world_size,
                local_rank=local_rank,
            )
    
        if distributed:
            try:
                train_loader.sampler.set_epoch(epoch)
            except AttributeError:
                train_loader.batch_sampler.sampler.set_epoch(epoch)

        for data, data_teacher in train_loader:
            for k in data.keys():
                data[k] = data[k].cuda()
                data_teacher[k] = data_teacher[k].cuda()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, y_pred_aux = model(data)
                    
                with torch.no_grad():
                    y_pred_teacher, y_pred_aux_teacher = model_teacher(data_teacher)

                loss = loss_fct(y_pred, y_pred_aux, data["target"], 0)
                loss += consistency_loss(
                    y_pred,
                    y_pred_teacher,
                    step=step,
                    student_pred_aux=y_pred_aux,
                    teacher_pred_aux=y_pred_aux_teacher,
                )

            scaler.scale(loss).backward()
            avg_losses.append(loss.detach())
            
#             if (
#                 optimizer_config['use_awp'] and
#                 (step % optimizer_config['awp_period']) == 0 and
#                 step > optimizer_config["awp_start_step"]
#             ):
#                 awp.perturb()
                
#             if optimizer_config['use_awp']:
#                 awp.restore()

            scaler.unscale_(optimizer)
            if optimizer_config["max_grad_norm"]:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    optimizer_config["max_grad_norm"],
                    # error_if_nonfinite=False,
                )
            scaler.step(optimizer)
            scale = scaler.get_scale()
            scaler.update()

            model.zero_grad(set_to_none=True)

            if distributed:
                torch.cuda.synchronize()

            update_teacher_params(model, model_teacher, mt_config["ema_decay"], step)

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

                if (step - 1) >= (epochs * len(train_loader) * 0.5):
                    preds_teach, _, _ = evaluate(
                        model_teacher,
                        val_loader,
                        loss_config,
                        loss_fct,
                        use_fp16=use_fp16,
                        distributed=distributed,
                        world_size=world_size,
                        local_rank=local_rank,
                    )
                else:
                    preds_teach = None

                if local_rank == 0:
                    preds = preds[: len(val_dataset)]
                    acc = accuracy(val_dataset.targets, preds.argmax(-1))

                    if preds_teach is not None:
                        preds_teach = preds_teach[: len(val_dataset)]
                        acc_teach = accuracy(val_dataset.targets, preds_teach.argmax(-1))

                    dt = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    step_ = step * world_size

                    s = f"Epoch {epoch:02d}/{epochs:02d} (step {step_:04d}) \t"
                    s = s + f"lr={lr:.1e} \t t={dt:.0f}s  \t loss={avg_loss:.3f}"
                    s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                    s = s + f"    acc={acc:.3f}" if acc else s
                    s = s + f"    acc_teach={acc_teach:.3f}" if acc_teach else s

                    print(s)

                if run is not None:
                    run[f"fold_{fold}/train/epoch"].log(epoch, step=step_)
                    run[f"fold_{fold}/train/loss"].log(avg_loss, step=step_)
                    run[f"fold_{fold}/train/lr"].log(lr, step=step_)
                    run[f"fold_{fold}/val/loss"].log(avg_val_loss, step=step_)
                    run[f"fold_{fold}/val/acc"].log(acc, step=step_)
                    if acc_teach:
                        run[f"fold_{fold}/val/acc_teach"].log(acc_teach, step=step_)

                start_time = time.time()
                avg_losses = []
                model.train()
                model_teacher.train()

        if (log_folder is not None) and (local_rank == 0) and model_soup:
            name = model.module.name if distributed else model.name
            if epoch >= epochs - 10:
                save_model_weights(
                    model.module if distributed else model,
                    f"{name.split('/')[-1]}_{fold}_{epoch}.pt",
                    cp_folder=log_folder,
                    verbose=0,
                )
                save_model_weights(
                    model_teacher,
                    f"{model_teacher.name.split('/')[-1]}_teacher_{fold}_{epoch}.pt",
                    cp_folder=log_folder,
                    verbose=0,
                )

    if (log_folder is not None) and (local_rank == 0):
        save_model_weights(
            model_teacher,
            f"{model_teacher.name.split('/')[-1]}_teacher_{fold}.pt",
            cp_folder=log_folder,
            verbose=0
        )

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    if distributed:
        torch.distributed.barrier()

    return preds

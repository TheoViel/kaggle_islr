import gc
import time
import torch
import numpy as np
# from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data.loader import define_loaders
from training.losses import SignLoss, ConsistencyLoss
from training.optim import define_optimizer, update_teacher_params, AWP
from model_zoo.utils import modify_drop
from utils.metrics import accuracy
from utils.torch import sync_across_gpus, save_model_weights


class Mixup(torch.nn.Module):
    """
    Mixup augmentation module for training neural networks.

    Attributes:
        beta_distribution (torch.distributions.Beta): The Beta distribution with the specified alpha parameter.
        num_classes (int): The number of classes in the classification task.

    Methods:
        forward(y): Performs Mixup augmentation on the input tensor.
    """
    def __init__(self, alpha=0.4):
        """
        Constructor
        
        Args:
            alpha (float, optional): The alpha parameter of the Beta distribution used for mixing coefficients.
                                     Defaults to 0.4.
        """
        super(Mixup, self).__init__()
        self.beta_distribution = torch.distributions.Beta(alpha, alpha)
        self.num_classes = 250

    def forward(self, y):
        """
        Performs Mixup augmentation on the input tensor.

        Args:
            y (torch.Tensor): The input tensor of shape (batch_size, num_classes) representing the ground truth labels.

        Returns:
            perm (torch.Tensor): A tensor of shape (batch_size,) representing the index permutation applied to the input tensor.
            coeffs (torch.Tensor): A tensor of shape (batch_size,) representing the mixing coefficients sampled from the Beta distribution.
            y (torch.Tensor): The augmented output tensor of shape (batch_size, num_classes).
        """
        bs = y.shape[0]
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(y.device)

        y = torch.zeros(y.size(0), self.num_classes).to(y.device).scatter(1, y.view(-1, 1).long(), 1)
        y = coeffs.view(-1, 1) * y + (1 - coeffs.view(-1, 1)) * y[perm]

        return perm, coeffs, y


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
    Evaluate the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for the validation set.
        loss_config (dict): Configuration parameters for the loss function.
        loss_fct (nn.Module): The loss function to compute the evaluation loss.
        use_fp16 (bool, optional): Whether to use mixed precision training. Defaults to False.
        distributed (bool, optional): Whether to use distributed training. Defaults to False.
        world_size (int, optional): Number of processes in distributed training. Defaults to 0.
        local_rank (int, optional): Local process rank in distributed training. Defaults to 0.

    Returns:
        preds (torch.Tensor or int): Predictions for the main task. If distributed training, 0 is returned for non-master processes.
        preds_aux (torch.Tensor or int): Predictions for auxiliary tasks. If distributed training, 0 is returned for non-master processes or if there are no auxiliary tasks.
        val_loss (float or int): Average validation loss. If distributed training, 0 is returned for non-master processes.
    """
        
    model.eval()
    val_losses = []
    preds, preds_aux = [], []
    
    try:
        num_classes_aux = model.module.num_classes_aux
    except AttributeError:
        num_classes_aux = model.num_classes_aux

    with torch.no_grad():
        for data, _, _ in val_loader:
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
    model_distilled,
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
    Train the model.

    Args:
        model (nn.Module): The main model to train.
        model_teacher (nn.Module): The teacher model for consistency regularization.
        model_distilled (nn.Module): The distilled model for additional regularization.
        train_dataset (Dataset): Dataset for training.
        val_dataset (Dataset): Dataset for validation.
        data_config (dict): Configuration parameters for data loading.
        loss_config (dict): Configuration parameters for the loss function.
        optimizer_config (dict): Configuration parameters for the optimizer.
        mt_config (dict): Configuration parameters for model teacher.
        epochs (int, optional): Number of training epochs. Defaults to 1.
        verbose_eval (int, optional): Number of steps for verbose evaluation. Defaults to 1.
        use_fp16 (bool, optional): Whether to use mixed precision training. Defaults to False.
        model_soup (bool, optional): Whether to save model weights during training. Defaults to False.
        distributed (bool, optional): Whether to use distributed training. Defaults to False.
        local_rank (int, optional): Local process rank in distributed training. Defaults to 0.
        world_size (int, optional): Number of processes in distributed training. Defaults to 1.
        log_folder (str, optional): Folder path for saving model weights. Defaults to None.
        run (neptune.Run, optional): Neptune run object for logging training progress. Defaults to None.
        fold (int, optional): Fold number for tracking progress. Defaults to 0.

    Returns:
        preds (torch.Tensor or int): Predictions for the main task. If distributed training, 0 is returned for non-master processes.
    """
    scaler = torch.cuda.amp.GradScaler()

    optimizer = define_optimizer(
        model,
        optimizer_config["name"],
        lr=optimizer_config["lr"],
        betas=optimizer_config["betas"],
        weight_decay=optimizer_config["weight_decay"],
    )
    
    if model_distilled is not None:
        optimizer_distilled = define_optimizer(
            model_distilled,
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
    if model_distilled is not None:
        scheduler_distilled = get_linear_schedule_with_warmup(
            optimizer_distilled, num_warmup_steps, num_training_steps
        )

    loss_fct = SignLoss(loss_config)

    mt_config["rampup_length"] = int(num_training_steps * mt_config["rampup_prop"])
    consistency_loss = ConsistencyLoss(mt_config)

    mix = Mixup(alpha=data_config['mix_alpha'])

    step = 1
    acc, acc_teach, acc_dist = 0, 0, 0
    preds_teach, preds_dist = None, None
    avg_losses = []
    start_time = time.time()
    for epoch in range(1, epochs + 1):

        if epoch == min(50, epochs // 2):
            train_dataset.aug_strength = min(train_dataset.aug_strength, 2)

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

        for data, data_teacher, data_distilled in train_loader:
            for k in data.keys():
                data[k] = data[k].cuda()
                data_teacher[k] = data_teacher[k].cuda()
                if model_distilled is not None:
                    data_distilled[k] = data_distilled[k].cuda()
                    
            perm, coefs = None, None
            y = data["target"]
            if mix is not None and np.random.random() < (1 - epoch / (0.9 * epochs)) * data_config['mix_proba']:
#                 if epoch >= epochs * optimizer_config["warmup_prop"]:
                perm, coefs, y = mix(data['target'])

            with torch.cuda.amp.autocast(enabled=use_fp16):
                y_pred, y_pred_aux = model(data, perm=perm, coefs=coefs)
                    
                with torch.no_grad():
                    y_pred_teacher, y_pred_aux_teacher = model_teacher(data_teacher, perm=perm, coefs=coefs)

                loss = loss_fct(y_pred, y_pred_aux, y, 0)

                loss += consistency_loss(
                    y_pred,
                    y_pred_teacher,
                    step=step,
                    student_pred_aux=y_pred_aux,
                    teacher_pred_aux=y_pred_aux_teacher,
                )
                
                if model_distilled is not None:
                    y_pred_dist, y_pred_aux_dist = model_distilled(data_distilled, perm=perm, coefs=coefs)
                    loss_dist = loss_fct(y_pred_dist, y_pred_aux_dist, y, 0) 
                    loss_dist += consistency_loss(y_pred_dist, y_pred_teacher, step)

            scaler.scale(loss).backward()
            if model_distilled is not None:
                scaler.scale(loss_dist).backward()
                
            avg_losses.append(loss.detach())

            scaler.unscale_(optimizer)
            if model_distilled is not None:
                scaler.unscale_(optimizer_distilled)
            
            if optimizer_config["max_grad_norm"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_config["max_grad_norm"])
                if model_distilled is not None:
                    torch.nn.utils.clip_grad_norm_(model_distilled.parameters(), optimizer_config["max_grad_norm"])

            scaler.step(optimizer)
            if model_distilled is not None:
                scaler.step(optimizer_distilled)

            scale = scaler.get_scale()
            scaler.update()

            model.zero_grad(set_to_none=True)
            if model_distilled is not None:
                model_distilled.zero_grad(set_to_none=True)

            if distributed:
                torch.cuda.synchronize()

            update_teacher_params(model, model_teacher, mt_config["ema_decay"], step)

            if scale == scaler.get_scale():
                scheduler.step()
                if model_distilled is not None:
                    scheduler_distilled.step()
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
                    if model_distilled is not None:
                        preds_dist, _, _ = evaluate(
                            model_distilled,
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
                    preds_dist = None

                if local_rank == 0:
                    preds = preds[: len(val_dataset)]
                    acc = accuracy(val_dataset.targets, preds.argmax(-1))

                    if preds_teach is not None:
                        preds_teach = preds_teach[: len(val_dataset)]
                        acc_teach = accuracy(val_dataset.targets, preds_teach.argmax(-1))
                    if preds_dist is not None:
                        preds_dist = preds_dist[: len(val_dataset)]
                        acc_dist = accuracy(val_dataset.targets, preds_dist.argmax(-1))

                    dt = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    step_ = step * world_size

                    s = f"Epoch {epoch:02d}/{epochs:02d} (step {step_:04d}) \t"
                    s = s + f"lr={lr:.1e} \t t={dt:.0f}s  \t loss={avg_loss:.3f}"
                    s = s + f"\t val_loss={avg_val_loss:.3f}" if avg_val_loss else s
                    s = s + f"    acc={acc:.3f}" if acc else s
                    s = s + f"    acc_teach={acc_teach:.3f}" if acc_teach else s
                    s = s + f"    acc_dist={acc_dist:.3f}" if acc_dist else s

                    print(s)

                if run is not None:
                    run[f"fold_{fold}/train/epoch"].log(epoch, step=step_)
                    run[f"fold_{fold}/train/loss"].log(avg_loss, step=step_)
                    run[f"fold_{fold}/train/lr"].log(lr, step=step_)
                    run[f"fold_{fold}/val/loss"].log(avg_val_loss, step=step_)
                    run[f"fold_{fold}/val/acc"].log(acc, step=step_)
                    if acc_teach:
                        run[f"fold_{fold}/val/acc_teach"].log(acc_teach, step=step_)
                    if acc_dist:
                        run[f"fold_{fold}/val/acc_dist"].log(acc_dist, step=step_)

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
                    f"{name.split('/')[-1]}_teacher_{fold}_{epoch}.pt",
                    cp_folder=log_folder,
                    verbose=0,
                )
                if model_distilled is not None:
                    save_model_weights(
                        model_distilled.module if distributed else model_distilled,
                        f"{name.split('/')[-1]}_distilled_{fold}_{epoch}.pt",
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
        if model_distilled is not None:
            name = f"{model_distilled.module.name.split('/')[-1]}_distilled_{fold}.pt"
            save_model_weights(model_distilled, name, cp_folder=log_folder, verbose=0)

    del (train_loader, val_loader, optimizer)
    torch.cuda.empty_cache()
    gc.collect()

    if distributed:
        torch.distributed.barrier()

    return preds

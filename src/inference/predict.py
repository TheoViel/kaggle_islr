import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from params import NUM_WORKERS

FLIPS = [None, [-1]]


def predict(model, dataset, loss_config, batch_size=64, device="cuda", use_fp16=False):
    """
    Torch predict function.

    Args:
        model (torch model): Model to predict with.
        dataset (CustomDataset): Dataset to predict on.
        loss_config (dict): Loss config, used for activation functions.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".
        use_fp16 (bool, optional): Whether to use fp16. Defaults to False.

    Returns:
        np array [n x n_classes]: Predictions.
        np array [n x n_classes_aux]: Aux predictions.
        np array [n x nb_fts]: Encoder features.
    """
    model.eval()
    preds = np.empty((0, model.num_classes))
    preds_aux = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for data, _ in tqdm(loader):
            for k in data.keys():
                data[k] = data[k].cuda()

            with torch.cuda.amp.autocast(enabled=use_fp16):
                pred, pred_aux = model(data)

            # Get probabilities
            if loss_config["activation"] == "sigmoid":
                pred = pred.sigmoid()
            elif loss_config["activation"] == "softmax":
                pred = pred.softmax(-1)

            if loss_config.get("activation_aux", "softmax") == "sigmoid":
                pred_aux = pred_aux.sigmoid()
            elif loss_config.get("activation_aux", "softmax") == "softmax":
                pred_aux = pred_aux.softmax(-1)

            preds = np.concatenate([preds, pred.cpu().numpy()])
            preds_aux.append(pred_aux.cpu().numpy())

    return preds, np.concatenate(preds_aux)


def predict_tta(
    model, dataset, loss_config, batch_size=64, device="cuda", use_fp16=False
):
    """
    Torch predict function with flip TTA.

    Args:
        model (torch model): Model to predict with.
        dataset (CustomDataset): Dataset to predict on.
        loss_config (dict): Loss config, used for activation functions.
        batch_size (int, optional): Batch size. Defaults to 64.
        device (str, optional): Device for torch. Defaults to "cuda".
        use_fp16 (bool, optional): Whether to use fp16. Defaults to False.

    Returns:
        np array [n x n_classes]: Predictions.
        np array [n x n_classes_aux]: Aux predictions.
        np array [n x nb_fts]: Encoder features.
    """
    model.eval()
    preds = np.empty((0, model.num_classes))
    preds_aux = np.empty((0, model.num_classes_aux))

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            preds_tta = []
            preds_tta_aux = []

            for f in FLIPS:
                # Forward
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    pred, pred_aux = model(torch.flip(x, f) if f is not None else x)

                # Get probabilities
                if loss_config["activation"] == "sigmoid":
                    pred = pred.sigmoid()
                elif loss_config["activation"] == "softmax":
                    pred = pred.softmax(-1)

                if loss_config.get("activation_aux", "softmax") == "sigmoid":
                    pred_aux = pred_aux.sigmoid()
                elif loss_config.get("activation_aux", "softmax") == "softmax":
                    pred_aux = pred_aux.softmax(-1)

                preds_tta.append(pred.cpu().numpy())
                preds_tta_aux.append(pred_aux.cpu().numpy())

            preds = np.concatenate([preds, np.mean(preds_tta, 0)])
            preds_aux = np.concatenate([preds_aux, np.mean(preds_tta_aux, 0)])

    return preds, preds_aux, None

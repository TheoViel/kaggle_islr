import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from params import NUM_WORKERS



def predict(model, dataset, loss_config, batch_size=64, device="cuda", use_fp16=False):
    """
    Perform model inference on a dataset.

    Args:
        model (nn.Module): Trained model for inference.
        dataset (Dataset): Dataset to perform inference on.
        loss_config (dict): Loss configuration.
        batch_size (int, optional): Batch size for inference. Defaults to 64.
        device (str, optional): Device to use for inference. Defaults to "cuda".
        use_fp16 (bool, optional): Flag indicating whether to use mixed precision inference. Defaults to False.

    Returns:
        preds (numpy.ndarray): Predicted probabilities of shape (num_samples, num_classes).
        preds_aux (numpy.ndarray): Auxiliary predictions of shape (num_samples, num_aux_classes).
    """
    model.eval()
    preds = np.empty((0, model.num_classes))
    preds_aux = []

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )

    with torch.no_grad():
        for data, _, _ in tqdm(loader):
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

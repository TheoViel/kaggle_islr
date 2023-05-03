import numpy as np


def accuracy(labels, predictions, beta=1):
    """
    Accuracy metric.

    Args:
        labels (np array [n]): Labels.
        predictions (np array [n] or [n x num_classes]): Predictions.

    Returns:
        float: Accuracy value.
    """
    labels = np.array(labels).squeeze()
    predictions = np.array(predictions).squeeze()

    if len(predictions.shape) > 1:
        predictions = predictions.argmax(-1)

    acc = (predictions == labels).mean()
    return acc

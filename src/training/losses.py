import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.consistency_weight = config["consistency_weight"]
        self.rampup_length = config["rampup_length"]  # rampup steps

    def sigmoid_rampup(self, current):
        if self.rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, self.rampup_length)
        phase = 1.0 - current / self.rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def get_consistency_weight(self, epoch):
        return self.consistency_weight * self.sigmoid_rampup(epoch)

    def forward(self, student_pred, teacher_pred, step):
        student_pred = student_pred.softmax(-1)
        teacher_pred = teacher_pred.softmax(-1).detach().data

        loss = ((student_pred - teacher_pred) ** 2).sum(-1)

        w = self.get_consistency_weight(step)

        return w * loss.mean()


class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    """

    def __init__(self, eps=0.0):
        """
        Constructor.
        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Computes the loss.
        Args:
            inputs (torch tensor [bs x n]): Predictions.
            targets (torch tensor [bs x n] or [bs]): Targets.
        Returns:
            torch tensor: Loss values, averaged.
        """
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)

        if self.eps > 0:
            n_class = inputs.size(1)
            targets = targets * (1 - self.eps) + (1 - targets) * self.eps / (
                n_class - 1
            )

        loss = -targets * F.log_softmax(inputs, dim=1)
        loss = loss.sum(-1)

        return loss


class SignLoss(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.ousm_k = config.get("ousm_k", 0)
        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(eps=self.eps)
        else:
            raise NotImplementedError

        self.loss_aux = nn.MSELoss(reduction="none")
        self.embed = torch.from_numpy(np.load("../output/embed.npy")).to(device)

    def prepare(self, pred, y):
        """
        Prepares the loss inputs.

        Args:
            pred (torch tensor  [bs x num_classes]): Predictions.
            y (torch tensor [bs] or [bs  x num_classes]): Target.

        Returns:
            _type_: _description_
        """
        if self.config["name"] in ["ce", "supcon"]:
            y = y.view(-1).long()

        else:
            y = y.float()
            pred = pred.float().view(y.size())

        if self.eps and self.config["name"] == "bce":
            y = torch.clamp(y, self.eps, 1 - self.eps)

        return pred, y

    def forward(self, pred, pred_aux, y, y_aux):
        """
        Computes the loss.

        Args:
            pred (torch tensor  [bs x num_classes]): Predictions.
            pred_aux (torch tensor [bs x num_classes_aux]): Aux predictions.
            y (torch tensor [bs] or [bs  x num_classes]): Target.
            y_aux (torch tensor [bs] or  [bs x num_classes_aux]): Aux target.

        Returns:
            torch tensor: Loss value, averaged.
        """
        pred, y = self.prepare(pred, y)

        loss = self.loss(pred, y)

        if self.ousm_k:
            _, idxs = loss.topk(y.size(0) - self.ousm_k, largest=False)
            loss = loss.index_select(0, idxs)

        loss = loss.mean()

        if not self.aux_loss_weight > 0:
            return loss

        y_aux = self.embed[y]
        loss_aux = self.loss_aux(pred_aux, y_aux).mean()

        return loss + self.aux_loss_weight * loss_aux

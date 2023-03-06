import torch
import torch.nn as nn
import torch.nn.functional as F


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

        return loss.mean()


class BreastLoss(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]

        self.eps = config.get("smoothing", 0)
        pos_weight = (
            torch.tensor([config["pos_weight"]]).to(device)
            if config["pos_weight"] is not None
            else None
        )

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
        elif config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(eps=self.eps)

        self.loss_aux = nn.CrossEntropyLoss(reduction="none")

    def prepare(self, pred, y):
        """
        Prepares the loss inputs.

        Args:
            pred (torch tensor  [bs x num_classes]): Predictions.
            y (torch tensor [bs] or [bs  x num_classes]): Target.

        Returns:
            _type_: _description_
        """
        if self.config["name"] == "ce":
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

        if not self.aux_loss_weight:
            return loss

        w_aux = (y_aux > -1).float().view(-1)
        y_aux = torch.clamp(y_aux, 0, 10000).view(-1).long()

        loss_aux = self.loss_aux(pred_aux, y_aux)
        loss_aux = (loss_aux * w_aux).sum() / (w_aux + 1e-6).sum()

        return loss + self.aux_loss_weight * loss_aux

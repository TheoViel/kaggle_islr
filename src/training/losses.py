import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """
    Consistency loss module for training a student-teacher model.

    Attributes:
        consistency_weight (float): Weight for the consistency loss.
        rampup_length (int): Length of the ramp-up period in steps.
        aux_loss_weight (float): Weight for the auxiliary loss.

    Methods:
        __init__(self, config): Constructor.
        sigmoid_rampup(current): Computes the sigmoid ramp-up value based on the current step.
        get_consistency_weight(epoch): Computes the consistency weight based on the current epoch.
        forward(student_pred, teacher_pred, step=1, student_pred_aux=None, teacher_pred_aux=None):
            Computes the consistency loss between student and teacher predictions.
    """
    def __init__(self, config):
        """
        Constructor.

        Args:
            config (dict): Configuration parameters.
                - consistency_weight (float): Weight for the consistency loss.
                - rampup_length (int): Length of the ramp-up period in steps.
                - aux_loss_weight (float, optional): Weight for the auxiliary loss. Defaults to 0.
        """
        super().__init__()
        self.consistency_weight = config["consistency_weight"]
        self.rampup_length = config["rampup_length"]  # rampup steps
        self.aux_loss_weight = config.get("aux_loss_weight", 0)

    def sigmoid_rampup(self, current):
        """
        Computes the sigmoid ramp-up value based on the current step.

        Args:
            current (float): Current step.

        Returns:
            float: Sigmoid ramp-up value.

        """
        if self.rampup_length == 0:
            return 1.0
        current = np.clip(current, 0.0, self.rampup_length)
        phase = 1.0 - current / self.rampup_length
        return float(np.exp(-5.0 * phase * phase))

    def get_consistency_weight(self, epoch):
        """
        Computes the consistency weight based on the current epoch.

        Args:
            epoch (int): Current epoch.

        Returns:
            float: Consistency weight.

        """
        return self.consistency_weight * self.sigmoid_rampup(epoch)

    def forward(
        self, student_pred, teacher_pred, step=1, student_pred_aux=None, teacher_pred_aux=None
    ):
        """
        Computes the consistency loss between student and teacher predictions.

        Args:
            student_pred (torch.Tensor): Predictions from the student model.
            teacher_pred (torch.Tensor): Predictions from the teacher model.
            step (int, optional): Current step. Defaults to 1.
            student_pred_aux (None or list of torch.Tensor, optional): Auxiliary predictions
                from the student model. Defaults to None.
            teacher_pred_aux (None or list of torch.Tensor, optional): Auxiliary predictions
                from the teacher model. Defaults to None.

        Returns:
            torch.Tensor: Consistency loss.

        """
        w = self.get_consistency_weight(step)

        student_pred = student_pred.softmax(-1)
        teacher_pred = teacher_pred.softmax(-1).detach().data
        loss = ((student_pred - teacher_pred) ** 2).sum(-1).mean()

        if not self.aux_loss_weight > 0:
            return w * loss.mean()

        w_aux_tot = 0
        loss_aux_tot = 0
        if isinstance(student_pred_aux, list):
            assert isinstance(teacher_pred_aux, list)

            for layer, (sp, tp) in enumerate(zip(student_pred_aux, teacher_pred_aux)):
                sp = sp.softmax(-1)
                tp = tp.softmax(-1).detach().data
                loss_aux = ((sp - tp) ** 2).sum(-1)

                loss_aux_tot += (self.aux_loss_weight * (layer + 1)) * loss_aux.mean()
                w_aux_tot += (self.aux_loss_weight * (layer + 1))

        return w * ((1 - w_aux_tot) * loss + w_aux_tot * loss_aux_tot)


class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Attributes:
        eps (float): Smoothing value.

    Methods:
        __init__(self, eps=0.0, device="cuda"): Constructor.
        forward(self, inputs, targets): Computes the loss.

    """
    def __init__(self, eps=0.0, device="cuda"):
        """
        Constructor.

        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
            device (str, optional): Device to use for computations. Defaults to "cuda".
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Computes the loss.

        Args:
            inputs (torch.Tensor): Predictions of shape [bs x n].
            targets (torch.Tensor): Targets of shape [bs x n] or [bs].

        Returns:
            torch.Tensor: Loss values, averaged.
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
    """
    Loss wrapper for the problem.

    Attributes:
        config (dict): Configuration parameters.
        device (str): Device to use for computations.
        aux_loss_weight (float): Weight for the auxiliary loss.
        ousm_k (int): Number of samples to exclude in the OUSM variant. Defaults to 0.
        eps (float): Smoothing value. Defaults to 0.
        loss (nn.Module): Loss function.
        loss_aux (nn.Module): Auxiliary loss function.

    Methods:
        __init__(self, config, device="cuda"): Constructor.
        prepare(self, pred, y): Prepares the predictions and targets for loss computation.
        forward(self, pred, pred_aux, y, y_aux): Computes the loss.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor.

        Args:
            config (dict): Configuration parameters.
            device (str, optional): Device to use for computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.ousm_k = config.get("ousm_k", 0)
        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(
                eps=self.eps, device=device
            )
        else:
            raise NotImplementedError

        self.loss_aux = nn.MSELoss(reduction="none")

    def prepare(self, pred, y):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Predictions.
            y (torch.Tensor): Targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] in ["ce", "supcon"]:
            y = y.squeeze()
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
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, y = self.prepare(pred, y)

        loss = self.loss(pred, y)

        if self.ousm_k:
            _, idxs = loss.topk(y.size(0) - self.ousm_k, largest=False)
            loss = loss.index_select(0, idxs)

        loss = loss.mean()

        if not self.aux_loss_weight > 0:
            return loss

        w_aux_tot = 0
        loss_aux_tot = 0
        if isinstance(pred_aux, list):
            for layer, p in enumerate(pred_aux):
                loss_aux = self.loss(p, y)

                if self.ousm_k:
                    loss_aux = loss_aux.index_select(0, idxs)

                loss_aux = loss_aux.mean()

                loss_aux_tot += (self.aux_loss_weight * (layer + 1)) * loss_aux
                w_aux_tot += (self.aux_loss_weight * (layer + 1))

        return (1 - w_aux_tot) * loss + w_aux_tot * loss_aux_tot

import torch
import numpy as np
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


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, temperature=1, contrast_mode='one', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels):
        """
        Compute loss for model.
        If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss.

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        if len(features.shape) == 2:
            features = features.unsqueeze(1)

        features = features.view(features.shape[0], features.shape[1], -1)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
            
#         print(anchor_feature.size(), contrast_feature.size())

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )
#         print(anchor_dot_contrast.size())
#         print(anchor_dot_contrast.max())
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        
#         print(logits.max(), logits.min())

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(features.device),
            0
        )
#         print(mask)
#         print(logits_mask)
        mask = mask * logits_mask
#         print(mask)
            
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
#         print("exp_logits", exp_logits.sum(1))
#         print(log_prob)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        
#         print("mask", mask.sum(1))
#         print("mean_log_prob_pos", mean_log_prob_pos)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
#         print(loss)
#         print()
#         return xczecz

        return loss


class SignLoss(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]

        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="mean")
        elif config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(eps=self.eps)
        elif config["name"] == "supcon":
#             print("supcon")
            self.loss = SupConLoss()

        self.loss_aux = nn.MSELoss(reduction="mean")
        self.embed = torch.from_numpy(np.load('../output/embed.npy')).to(device)

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

        if not self.aux_loss_weight > 0:
            return loss

        y_aux = self.embed[y]
        loss_aux = self.loss_aux(pred_aux, y_aux)

        return loss + self.aux_loss_weight * loss_aux

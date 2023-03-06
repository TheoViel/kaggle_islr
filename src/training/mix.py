import torch
import numpy as np
import torch.nn as nn


def rand_bbox(size, lam):
    """
    Retuns the coordinate of a random rectangle in the image for cutmix.

    Args:
        size (torch tensor [batch_size x c x W x H): Input size.
        lam (int): Lambda sampled by the beta distribution. Controls the size of the squares.

    Returns:
        int: 4 coordinates of the rectangle.
        int: Proportion of the unmasked image.
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return bbx1, bby1, bbx2, bby2, lam


def cutmix_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies cutmix to a sample.

    Args:
        x (torch tensor [batch_size x input_size]): Input batch.
        y (torch tensor [batch_size]): Labels.
        alpha (float, optional): Parameter of the beta distribution. Defaults to 1.0.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        torch tensor [batch_size x input_size]: Mixed input.
        torch tensor [batch_size x num_classes]: Mixed labels.
        float: Probability sampled by the beta distribution.
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    index = torch.randperm(x.size()[0]).to(device)

    bbx1, bby1, bbx2, bby2, lam = rand_bbox(x.size(), lam)

    mixed_x = x.clone()
    mixed_x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    y_a, y_b = y, y[index]
    mixed_y = lam * y_a + (1 - lam) * y_b

    return mixed_x, mixed_y, lam


class Mixup(nn.Module):
    """
    Mixup Wrapper.
    """
    def __init__(self, alpha, additive=False):
        """
        Constructor.

        Args:
            alpha (float): Mixup alpha.
            additive (bool, optional): Whether to use additive mixup. Defaults to False.
        """
        super(Mixup, self).__init__()
        self.beta_distribution = torch.distributions.Beta(alpha, alpha)
        self.additive = additive

    def forward(self, x, y, y_aux):
        """
        Applies mixup.

        Args:
            x (torch tensor [bs x ...]): Model input.
            y (torch tensor [bs x ...]): Target.
            y_aux (torch tensor [bs x ...]): Aux target.

        Returns:
            torch tensor [bs x ...]): Mixed model input.
            torch tensor [bs x ...]): Mixed target.
            torch tensor ([bs x ...]): Mixed aux target.
        """
        bs = x.shape[0]
        n_dims = len(x.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(x.device)

        if n_dims == 2:
            x = coeffs.view(-1, 1) * x + (1 - coeffs.view(-1, 1)) * x[perm]
        elif n_dims == 3:
            x = coeffs.view(-1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1)) * x[perm]
        elif n_dims == 4:
            x = coeffs.view(-1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1)) * x[perm]
        else:
            x = (
                coeffs.view(-1, 1, 1, 1, 1) * x
                + (1 - coeffs.view(-1, 1, 1, 1, 1)) * x[perm]
            )

        if self.additive:
            y = (y + y[perm]).clip(0, 1)
            y_aux = (y_aux + y_aux[perm]).clip(0, 1)
        else:
            if len(y.shape) == 1:
                y = coeffs * y + (1 - coeffs) * y[perm]
                y_aux = coeffs * y_aux + (1 - coeffs) * y_aux[perm]
            else:
                y = coeffs.view(-1, 1) * y + (1 - coeffs.view(-1, 1)) * y[perm]
                y_aux = (
                    coeffs.view(-1, 1) * y_aux + (1 - coeffs.view(-1, 1)) * y_aux[perm]
                )

        return x, y, y_aux

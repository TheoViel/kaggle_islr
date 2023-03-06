import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class BreastDataset(Dataset):
    """
    Image torch Dataset.
    """

    def __init__(
        self,
        df,
        transforms=None,
    ):
        """
        Constructor

        Args:
            df (pd DataFrame): DataFrame.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.paths = df["path"].values
        self.transforms = transforms
        self.targets = df["cancer"].values
        self.targets_aux = df["BIRADS"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Item accessor.

        Args:
            idx (int): Index.

        Returns:
            torch tensor [C x H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Aux label.
        """
        image = cv2.imread(self.paths[idx], cv2.IMREAD_ANYDEPTH)
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        else:
            image = image.astype(np.float32) / 255.0

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if len(image.size()) == 2:
            image = torch.stack([image, image, image], 0)
        elif image.size(0) == 1:
            image = torch.cat([image, image, image], 0)

        y = torch.tensor([self.targets[idx]], dtype=torch.float)
        y_aux = torch.tensor([self.targets_aux[idx]], dtype=torch.float)

        return image, y, y_aux


def get_size(config):
    x = re.sub("[a-z]+", "", config.img_folder.lower()[:-1])
    x = re.sub("_", " ", x)
    x = re.sub(r"\s+", " ", x).strip().split()
    x = tuple(map(int, x))

    if len(x) == 1:
        x = (x[0], x[0])
    return x

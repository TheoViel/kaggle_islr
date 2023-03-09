import re
import torch
import numpy as np
from torch.utils.data import Dataset

from data.transforms import augment


def crop_or_pad(data, max_len=100):
    diff = max_len - data['x'].shape[0]
    
    if diff <= 0:  # Crop
        return {k : data[k][:max_len] for k in data.keys()}

    padding = torch.ones((diff, data['x'].shape[1]))
    
    padded = {}
    for k in data.keys():
        coef = -10 if k in ['x', 'y', 'z'] else 0
        padded[k] = torch.cat([data[k], coef * padding], axis=0).type(data[k].type())
        
    return padded


class SignDataset(Dataset):
    """
    Sign torch Dataset.
    """

    def __init__(
        self,
        df,
        max_len=None,
        train=False,
    ):
        """
        Constructor

        Args:
            df (pd DataFrame): DataFrame.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.paths = df["processed_path"].values
        self.targets = df["target"].values
        self.max_len = max_len
        self.train = train

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
        frames = np.load(self.paths[idx])
        
        landmark_embed = np.arange(frames.shape[-1])[None] + 1
        landmark_embed = np.repeat(landmark_embed, frames.shape[0], axis=0)        
        
        data = {
            "type": torch.tensor(frames[:, 0], dtype=torch.long),
            "landmark": torch.tensor(landmark_embed, dtype=torch.long),
            "x": torch.tensor(frames[:, 1], dtype=torch.float),
            "y": torch.tensor(frames[:, 2], dtype=torch.float),
            "z": torch.tensor(frames[:, 3], dtype=torch.float),
        }
        data["mask"] = torch.where(data["x"].clone() == -10, 1, 1)  # .bool()

        if self.train:
            data = augment(data)

        if self.max_len is not None:
            data = crop_or_pad(data, max_len=self.max_len)
        
        data["target"] = torch.tensor([self.targets[idx]], dtype=torch.float)

        return data

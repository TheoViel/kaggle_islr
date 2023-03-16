import re
import torch
import numpy as np
from torch.utils.data import Dataset

from data.transforms import augment, normalize
from params import TYPE_MAPPING


def crop_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data['x'].shape[0]

    if diff <= 0:  # Crop
        if mode == "start":
            return {k : data[k][:max_len] for k in data.keys()}
        else:
            offset = np.abs(diff) // 2
            return {k : data[k][offset: offset + max_len] for k in data.keys()}

    padding = torch.ones((diff, data['x'].shape[1]))
    
    padded = {}
    for k in data.keys():
        coef = 0  # -10 if k in ['x', 'y', 'z'] else 0
        padded[k] = torch.cat([data[k], coef * padding], axis=0).type(data[k].type())
        
    return padded


def regroup(d):
    to_regroup = ["left_eye", "left_eyebrow", "right_eye", "right_eyebrow", "nose"]

    for k in to_regroup:
        types = d[0, 0]
        dt = d.T[types == TYPE_MAPPING[k]].T
        d = d.T[types != TYPE_MAPPING[k]].T
        d = np.concatenate([d, dt.mean(-1, keepdims=True)], -1)

    return d


class SignDataset(Dataset):
    """
    Sign torch Dataset.
    """

    def __init__(
        self,
        df,
        max_len=None,
        aug_strength=0,
        train=False,
    ):
        """
        Constructor

        Args:
            df (pd DataFrame): DataFrame.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.max_len = max_len
        self.aug_strength = aug_strength
        self.train = train
        
        self.paths = df["processed_path"].values
        self.targets = df["target"].values

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
        
        if "processed_3/" in self.paths[idx]:
            frames = regroup(frames)

        landmark_embed = np.arange(frames.shape[-1])[None] + 1
        landmark_embed = np.repeat(landmark_embed, frames.shape[0], axis=0)        
        
        data = {
            "type": torch.tensor(frames[:, 0], dtype=torch.long),
            "landmark": torch.tensor(landmark_embed, dtype=torch.long),
            "x": torch.tensor(frames[:, 1], dtype=torch.float),
            "y": torch.tensor(frames[:, 2], dtype=torch.float),
            "z": torch.tensor(frames[:, 3], dtype=torch.float),
        }
        data["mask"] = torch.ones(data["x"].size())

        if "processed_3/" in self.paths[idx]:
            data = normalize(data)
            
        if self.train:
            data = augment(data, aug_strength=self.aug_strength)

        if self.max_len is not None:
            data = crop_or_pad(data, max_len=self.max_len)

        data["target"] = torch.tensor([self.targets[idx]], dtype=torch.float)

        return data

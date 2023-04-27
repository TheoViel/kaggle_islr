import copy
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from data.transforms import (
    augment,
    normalize,
    # interpolate,
    flip,
    add_missing_hand,
    # normalize_face,
)
# from params import TYPE_MAPPING


def crop_or_pad(data, max_len=100, mode="start"):
    diff = max_len - data["x"].shape[0]

    if diff <= 0:  # Crop
        if mode == "start":
            return {k: data[k][:max_len] for k in data.keys()}
        else:
            offset = np.abs(diff) // 2
            return {k: data[k][offset: offset + max_len] for k in data.keys()}

    padding = torch.ones((diff, data["x"].shape[1]))

    padded = {}
    for k in data.keys():
        if k in ["target", "length"]:
            padded[k] = data[k]
        else:
            coef = 0  # -10 if k in ['x', 'y', 'z'] else 0
            padded[k] = torch.cat([data[k], coef * padding], axis=0).type(
                data[k].type()
            )

    return padded


def resize(data, size=50):
    for k in ["x", "y", "z"]:
        data[k] = torch.nn.functional.interpolate(
            data[k].T.unsqueeze(0), size, mode="linear"
        )[0].T

    if "type" in data.keys():
        data["type"] = data["type"][:1].repeat(size, 1)

    return data


def is_left(data):
    fts = torch.abs(data["x"])
    embed = data["type"][0]

    left_hand = fts.T[embed == 1].T
    right_hand = fts.T[embed == 2].T

    missing_left = (left_hand.sum(-1) == 0).sum()
    missing_right = (right_hand.sum(-1) == 0).sum()

    return missing_left < missing_right


class SignDataset(Dataset):
    """
    Sign torch Dataset.
    """

    def __init__(
        self,
        df,
        max_len=None,
        aug_strength=0,
        resize_mode="pad",
        train=False,
        dist=False,
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
        self.dist = dist
        self.resize_mode = resize_mode

        self.paths = df["processed_path"].values
        self.targets = df["target"].values

        self.buffer = {}
        self.buffer_mode = False
        self.lens = {}

    def __len__(self):
        return len(self.paths)

    def fill_buffer(self, tqdm_enabled=False):
        self.buffer_mode = True
        loader = DataLoader(
            self,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            collate_fn=list,
            drop_last=False,
        )
        for i, batch in enumerate(tqdm(loader, disable=not tqdm_enabled)):
            for j, b in enumerate(batch):
                self.buffer[i * 1024 + j] = b

        self.buffer_mode = False

    def mix_face(self, data, idx, p=0):
        if np.random.random() > p:
            return data

        # Load other sample
        other_idx = np.random.choice(
            list(self.df[self.df["target"] == self.targets[idx]].index)
        )

        try:
            frames = self.buffer[other_idx]
        except KeyError:
            frames = np.load(self.paths[other_idx])

        # Create other data
        landmark_embed = np.arange(frames.shape[-1])[None] + 1
        landmark_embed = np.repeat(landmark_embed, frames.shape[0], axis=0)

        other_data = {
            "type": torch.tensor(frames[:, 0], dtype=torch.long),
            "landmark": torch.tensor(landmark_embed, dtype=torch.long),
            "x": torch.tensor(frames[:, 1], dtype=torch.float),
            "y": torch.tensor(frames[:, 2], dtype=torch.float),
            "z": torch.tensor(frames[:, 3], dtype=torch.float),
        }

        # Flip to same direction
        if is_left(other_data) != is_left(data):
            other_data = flip(other_data)

        # Resize to same size
        other_data = resize(other_data, data["x"].size(0))

        # Replace face
#         ids = torch.isin(data["type"][0], torch.tensor([4]))
#         replaced = torch.isin(data["type"], torch.tensor([4]))
#         ids = torch.isin(data["type"][0], torch.tensor([3, 6, 7, 8, 9, 10, 11]))
#         replaced = torch.isin(data["type"], torch.tensor([3, 6, 7, 8, 9, 10, 11]))
        ids = torch.isin(data["type"][0], torch.tensor([3, 4, 6]))
        replaced = torch.isin(data["type"], torch.tensor([3, 4, 6]))

        for k in ["x", "y"]:
            new_face = other_data[k].T[ids].T
            if (new_face.max() - new_face.min()) == 0:
                return data  # do not apply

        for k in ["x", "y", "z"]:
            new_face = other_data[k].T[ids].T
            old_face = data[k].T[ids].T

            if (k == "z") and ((new_face.max() - new_face.min()) == 0):
                return data

            other_data[k] = (other_data[k] - new_face.min()) / (
                new_face.max() - new_face.min()
            )
            other_data[k] = (
                other_data[k] * (old_face.max() - old_face.min()) + old_face.min()
            )
            data[k] = torch.where(replaced, other_data[k], data[k])

        return data

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
        try:
            frames = self.buffer[idx]
        except KeyError:
            frames = np.load(self.paths[idx])
            self.buffer[idx] = frames

        if self.buffer_mode:
            return frames

        length = len(frames)
        self.lens[idx] = length

        landmark_embed = np.arange(frames.shape[-1])[None] + 1
        landmark_embed = np.repeat(landmark_embed, frames.shape[0], axis=0)

        data = {
            "type": torch.tensor(frames[:, 0], dtype=torch.long),
            "landmark": torch.tensor(landmark_embed, dtype=torch.long),
            "x": torch.tensor(frames[:, 1], dtype=torch.float),
            "y": torch.tensor(frames[:, 2], dtype=torch.float),
            "z": torch.tensor(frames[:, 3], dtype=torch.float),
        }

        data["target"] = torch.tensor([self.targets[idx]], dtype=torch.float)

        data_mt, data_dist = 0, 0
        if self.train:
            data_mt = copy.deepcopy(data)
            if self.dist:
                data_dist = copy.deepcopy(data)

            if self.aug_strength >= 3:
                data = self.mix_face(data, idx, p=0.25)
                data = add_missing_hand(data, p=0.25)
                data_mt = self.mix_face(data_mt, idx, p=0.25)
                data_mt = add_missing_hand(data_mt, p=0.25)
                if self.dist:
                    data_dist = self.mix_face(data_dist, idx, p=0.25)
                    data_dist = add_missing_hand(data_dist, p=0.25)

            data = augment(data, aug_strength=self.aug_strength)

            data_mt = augment(data_mt, aug_strength=self.aug_strength)
            data_mt["mask"] = torch.ones(data_mt["x"].size())

            if self.dist:
                data_dist = augment(data_dist, aug_strength=self.aug_strength)
                data_dist["mask"] = torch.ones(data_dist["x"].size())

#                 data_dist = normalize(data_dist)
#             data_mt = normalize(data_mt)
#         data = normalize(data)

        data["mask"] = torch.ones(data["x"].size())
        
        if self.max_len is not None:
            if self.resize_mode == "pad":
                data = crop_or_pad(data, max_len=self.max_len)
                if self.train:
                    data_mt = crop_or_pad(data_mt, max_len=self.max_len)
                    if self.dist:
                        data_dist = crop_or_pad(data_dist, max_len=self.max_len)
            else:
                data = resize(data, size=self.max_len)
                if self.train:
                    data_mt = resize(data_mt, size=self.max_len)
                    if self.dist:
                        data_dist = resize(data_dist, size=self.max_len)

        return data, data_mt, data_dist

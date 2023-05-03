import scipy
import torch
import numpy as np


FLIP_MAP = {  # torch_8+
    53: 44, 52: 43, 56: 48, 59: 51, 54: 45, 57: 49, 55: 46, 58: 50, 47: 47,  # head
    41: 20, 40: 19, 39: 18, 38: 17, 37: 16, 36: 15, 35: 14, 34: 13, 33: 12, 32: 11, 31: 10, 30: 9,
    29: 8, 28: 7, 27: 6, 26: 5, 25: 4, 24: 3, 23: 2, 22: 1, 21: 0,  # hands
    97: 98, 95: 96, 94: 93,  # eyebrows, cheeks, eyes
    63: 73, 62: 72, 61: 71, 71: 61, 72: 62, 73: 63, 79: 69, 76: 66, 66: 76, 69: 79,  # outter lips
    64: 74, 70: 80, 65: 75, 78: 68, 67: 77,  # inner lips
    81: 82, 83: 84, 85: 86, 87: 88, 89: 90, 91: 92,  # arms
}


FLIP_ARRAY = np.arange(np.max(np.concatenate(list(FLIP_MAP.items()))) + 1)
for k, v in FLIP_MAP.items():
    FLIP_ARRAY[k] = v
    FLIP_ARRAY[v] = k


def flip(data, flip_array=FLIP_ARRAY, p=1):
    """
    Flips the data horizontally.

    Args:
        data (dict): The input data.
        flip_array (np.ndarray, optional): Maps ids to the flipped id. Defaults to FLIP_ARRAY_HANDS.
        p (float, optional): Probability of flipping the data. Defaults to 1.

    Returns:
        dict: The flipped data.
    """
    if np.random.random() > p:
        return data

    flip_array_ = np.arange(len(data["x"].T))
    flip_array = flip_array[: len(flip_array_)]
    flip_array_[: len(flip_array)] = flip_array

    data["x"] = - data["x"]
    for k in ["x", "y", "z"]:
        data[k] = data[k].T[flip_array_].T

    return data


FLIP_MAP_HANDS = {
    41: 20,
    40: 19,
    39: 18,
    38: 17,
    37: 16,
    36: 15,
    35: 14,
    34: 13,
    33: 12,
    32: 11,
    31: 10,
    30: 9,
    29: 8,
    28: 7,
    27: 6,
    26: 5,
    25: 4,
    24: 3,
    23: 2,
    22: 1,
    21: 0,  # hands
}

FLIP_ARRAY_HANDS = np.arange(np.max(np.concatenate(list(FLIP_MAP_HANDS.items()))) + 1)
for k, v in FLIP_MAP_HANDS.items():
    FLIP_ARRAY_HANDS[k] = v
    FLIP_ARRAY_HANDS[v] = k


def add_missing_hand(data, flip_array=FLIP_ARRAY_HANDS, p=1.0):
    """
    Adds missing hand data to the input dictionary.

    Args:
        data (dict): The input data.
        flip_array (np.ndarray, optional): Maps ids to the flipped id. Defaults to FLIP_ARRAY_HANDS.
        p (float, optional): Probability of adding missing hand data. Defaults to 1.0.

    Returns:
        dict: The data with missing hand data added.
    """
    if np.random.random() > p:
        return data

    flip_array_ = np.arange(len(data["x"].T))
    flip_array = flip_array[: len(flip_array_)]
    flip_array_[: len(flip_array)] = flip_array

    flipped = {}
    for k in ["x", "y", "z"]:
        flipped[k] = data[k].T[flip_array_].T
    flipped["x"] = -flipped["x"]

    shift = data["x"][:, -1].unsqueeze(-1)  # x_nose
    flipped["x"] += 2 * shift

    fts = torch.abs(data["x"])
    embed = data["type"][0]

    left_hand = fts.T[embed == 5].T
    right_hand = fts.T[embed == 10].T

    missing_left = left_hand.sum(-1) == 0
    missing_right = right_hand.sum(-1) == 0

    for k in ["x", "y", "z"]:
        data[k][missing_left] = data[k][missing_left] * (embed != 2).unsqueeze(
            0
        ) + flipped[k][missing_left] * (embed == 2).unsqueeze(0)
        data[k][missing_right] = data[k][missing_right] * (embed != 1).unsqueeze(
            0
        ) + flipped[k][missing_right] * (embed == 1).unsqueeze(0)

    return data


def scale(data, factor=0.3, p=0.5):
    """
    Scales the input data by a random factor.

    Args:
        data (dict): The input data.
        factor (float, optional): The scaling factor. Defaults to 0.3.
        p (float, optional): Probability of applying the scaling. Defaults to 0.5.

    Returns:
        dict: The scaled data.
    """
    if np.random.random() > p:
        return data

    distort = np.random.random() < p
    scale_factor = np.random.uniform(1 - factor, 1 + factor)

    for k in ["x", "y", "z"]:
        distort_factor = np.random.uniform(1 - factor, 1 + factor) if distort else 0
        data[k] *= scale_factor + distort_factor

    return data


def rotate(data, max_angle=1/6, p=0.5):
    """
    Rotates the input data by a random angle.

    Args:
        data (dict): The input data.
        max_angle (float, optional): The maximum angle in radians/2pi for rotation. Defaults to 1/6.
        p (float, optional): Probability of applying rotation. Defaults to 0.5.

    Returns:
        dict: The rotated data.
    """
    if np.random.random() > p:
        return data

    x_o = 0  # torch.randn(1) + data['x'].mean()
    y_o = 0  # torch.randn(1) + data['y'].max()

    angle = max_angle * 2 * np.pi * (torch.rand(1) - 0.5)  # [+/- max_angle]

    cos = np.cos(angle)
    sin = np.sin(angle)

    x = x_o + cos * (data["x"] - x_o) - sin * (data["y"] - y_o)
    y = y_o + sin * (data["x"] - x_o) + cos * (data["y"] - y_o)

    data["x"] = x
    data["y"] = y
    return data


def interpolate(data, p=0.5):
    """
    Interpolates missing values in the input data.

    Args:
        data (dict): The input data.
        p (float, optional): Probability of applying interpolation. Defaults to 0.5.

    Returns:
        dict: The data with missing values interpolated.
    """
    if np.random.random() > p:
        return data

    for k in ["x", "y", "z"]:
        for i in range(data[k].size(-1)):
            x = data[k][:, i]
            ids = torch.where(x != 0)[0]

            if len(ids) > 1 and (len(ids) < data[k].size(0)):
                x = x[ids]
                f = scipy.interpolate.interp1d(ids.numpy(), x.numpy())

                ids_interp = np.clip(np.arange(data[k].size(0)), ids.min(), ids.max())
                interp = torch.from_numpy(f(ids_interp))

                data[k][:, i] = interp

    return data


def crop(data, max_crop=0.2, p=0.5):
    """
    Randomly crops the input data.

    Args:
        data (dict): The input data.
        max_crop (float, optional): Maximum proportion of data to be cropped. Defaults to 0.2.
        p (float, optional): Probability of applying cropping. Defaults to 0.5.

    Returns:
        dict: The cropped data.
    """
    if np.random.random() > p:
        return data

    if len(data["x"]) < 20:  # too short
        return data

    crop = np.random.randint(1, int(len(data["x"]) * max_crop))
    if np.random.random() > 0.5:  # left
        return {
            k: (data[k][crop:] if k != "target" else data[k])
            for k in data.keys()
        }
    else:
        return {
            k: (data[k][:-crop] if k != "target" else data[k])
            for k in data.keys()
        }


def augment(data, aug_strength=0):
    """
    Augments the input data based on the specified augmentation strength.

    Args:
        data (dict): The input data.
        aug_strength (int): The augmentation strength level (0, 1, 2, or 3).

    Returns:
        dict: The augmented data.
    """
    if aug_strength == 3:
        data = interpolate(data, p=0.5)
        data = flip(data, p=0.5)
        data = rotate(data, p=0.5)
        data = crop(data, p=0.5)
        data = scale(data, p=0.25)

    if aug_strength == 2:
        data = interpolate(data, p=0.5)
        data = flip(data, p=0.5)
        data = rotate(data, p=0.5)
        data = scale(data, p=0.25)
        data = crop(data, p=0.25)

    elif aug_strength == 1:

        data = flip(data, p=0.5)
        data = rotate(data, max_angle=1 / 12, p=0.25)

    return data

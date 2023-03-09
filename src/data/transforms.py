import torch
import numpy as np


def drop_frames(data, prop=0.1, p=0.5):
    if np.random.random() > p:
        return data

    frames = np.arange(len(data['x']))
    to_keep = np.random.choice(frames, int((1 - prop) * len(frames)), replace=False)
    return {k: data[k][np.sort(to_keep)] for k in data.keys()}


def add_noise(data, snr=10, p=0.5):
    if np.random.random() > p:
        return data

    for k in ['x', 'y', 'z']:
        noise = torch.from_numpy(np.random.normal(scale=data[k].std() / snr, size=data[k].shape))
        data[k] += noise
    
    return data


def shift(data, snr=3, p=0.5):
    if np.random.random() > p:
        return data
    
    for k in ['x', 'y', 'z']:
        s = np.random.random() / ((data[k].max() - data[k].min()) / snr)
        data[k] += s * np.random.choice([-1, 1])
    
    return data


def augment(data):
#     data = drop_frames(data, p=0.3)
    data = add_noise(data, p=0.1)
    data = shift(data, p=0.3)
    return data

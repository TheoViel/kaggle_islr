import torch
import numpy as np


def normalize(data):
    for k in ['x', 'y', 'z']:
        x = data[k].flatten()
        x = x[x > -10]
        mean = x.mean()
        std = x.std(unbiased=False)

        data[k] = torch.where(
            data[k] > -10,
            (data[k] - mean) / (std + 1e-6),
            0,
        )
    return data
        

def scale(data, factor=0.3, p=0.5):
    if np.random.random() > p:
        return data
    
    distort = np.random.random() < p
    scale_factor = np.random.uniform(1 - factor, 1 + factor)

    for k in ['x', 'y', 'z']:
        distort_factor = np.random.uniform(1 - factor, 1 + factor) if distort else 0
        data[k] *= (scale_factor + distort_factor)
        
    return data

def dropout(data, drop_p=0.1, p=0.5):
    if np.random.random() > p:
        return data

    mask = torch.rand(data['x'].size()) > drop_p
    for k in ['x', 'y', 'z']:
        data[k] *= mask

    return data


def rotate(data, max_angle=1/6, p=0.5):
    if np.random.random() > p:
        return data

    x_o = 0  # torch.randn(1) + data['x'].mean()
    y_o = 0  # torch.randn(1) + data['y'].max()

    angle = max_angle * 2 * np.pi * (torch.rand(1) - 0.5)  # [+/- max_angle]
    
    cos = np.cos(angle)
    sin = np.sin(angle)
    
    x = x_o + cos * (data['x'] - x_o) - sin * (data['y'] - y_o)
    y = y_o + sin * (data['x'] - x_o) + cos * (data['y'] - y_o)
    
    data['x'] = x
    data['y'] = y
    return data
    
    
def drop_frames(data, prop=0.1, p=0.5):
    if np.random.random() > p:
        return data

    if len(data['x']) < 10:  # too short
        return data

    frames = np.arange(len(data['x']))
    to_keep = np.random.choice(frames, int((1 - prop) * len(frames)), replace=False)
    return {k: data[k][np.sort(to_keep)] for k in data.keys()}


def add_noise(data, snr=50, p=0.5):
    if np.random.random() > p:
        return data

    for k in ['x', 'y', 'z']:
        noise = torch.from_numpy(np.random.normal(scale=data[k].std() / snr, size=data[k].shape))
        data[k] += noise * (data[k] != 0)
    
    return data


def shift(data, snr=3, p=0.5):
    if np.random.random() > p:
        return data
    
    for k in ['x', 'y', 'z']:
        s = np.random.random() / ((data[k].max() - data[k].min()) / snr)
        data[k] += s * np.random.choice([-1, 1])
    
    return data


def augment(data, aug_strength=3):
    if aug_strength == 3:
        data = shift(data, p=0.75)
        data = scale(data, p=0.75)
        data = rotate(data, p=0.75)
        data = drop_frames(data, p=0.25)
        data = dropout(data, drop_p=0.2, p=0.5)
        data = add_noise(data, snr=3, p=0.5)

    if aug_strength == 2:
        data = shift(data, p=0.5)
        data = scale(data, p=0.5)
        data = rotate(data, p=0.5)
        data = dropout(data, p=0.25)
        data = add_noise(data, p=0.1)

    elif aug_strength == 1:
#         data = shift(data, p=0.5)
        data = rotate(data, p=0.5)
        data = scale(data, p=0.25)
#         data = add_noise(data, p=0.25, snr=50)
#         data = dropout(data, p=0.5)
        
    return data

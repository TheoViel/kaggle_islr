import scipy
import torch
import numpy as np


FLIP_MAP = {  # torch_3
    6: 26, 4: 24, 16: 35, 21: 40, 13: 33, 17: 36, 14: 34, 18: 37,  # silhouette
    20: 39, 98: 96, 97: 95, 99: 99,  # face
    41: 74, 42: 75, 43: 76, 44: 77, 45: 78, 46: 79, 47: 80, 48: 81, 49: 82, 50: 83, 51: 84, 52: 85, 53: 86, 54: 87, 55: 88, 56: 89, 57: 90, 58: 91, 59: 92, 60: 93, 61: 94,  # hands
    62: 63, 64: 65, 66: 67, 68: 69, 70: 71, 72: 73,  # pose
    2: 22, 3: 23, 9: 29, 12 : 32, 5: 25,  # outter lips
    27 : 7, 11: 31, 10: 30, 19: 38, 8: 1,  # intter lips
}

# FLIP_MAP = {  # torch_7
#     90: 81, 98: 89, 95: 86, 94: 85, 97: 88, 91: 82, 93: 84, 92: 83, 96: 87,  # eyes
#     53: 44, 52: 43, 56: 48, 59: 51, 54: 45, 57: 49, 55: 46, 58: 50, 47: 47,  # head
#     41: 20, 40: 19, 39: 18, 38: 17, 37: 16, 36: 15, 35: 14, 34: 13, 33: 12, 32: 11, 31: 10, 30: 9, 29: 8, 28: 7, 27: 6, 26: 5, 25: 4, 24: 3, 23: 2, 22: 1, 21: 0,  # hands
#     113: 114, 112: 111,  # eyebrows, cheeks
#     63: 73, 62: 72, 61: 71, 71: 61, 72: 62, 73: 63, 79: 69, 76: 66, 66: 76, 69: 79,  # outter lips
#     64: 74, 70: 80, 65: 75, 78: 68, 67: 77,  # inner lips
#     99: 100, 101: 102, 103: 104, 105: 106, 107: 108, 109: 110,  # arms
# }

FLIP_MAP = {  # torch_8+
    53: 44, 52: 43, 56: 48, 59: 51, 54: 45, 57: 49, 55: 46, 58: 50, 47: 47,  # head
    41: 20, 40: 19, 39: 18, 38: 17, 37: 16, 36: 15, 35: 14, 34: 13, 33: 12, 32: 11, 31: 10, 30: 9, 29: 8, 28: 7, 27: 6, 26: 5, 25: 4, 24: 3, 23: 2, 22: 1, 21: 0,  # hands
    97: 98, 95: 96, 94: 93,  # eyebrows, cheeks, eyes
    63: 73, 62: 72, 61: 71, 71: 61, 72: 62, 73: 63, 79: 69, 76: 66, 66: 76, 69: 79,  # outter lips
    64: 74, 70: 80, 65: 75, 78: 68, 67: 77,  # inner lips
    81: 82, 83: 84, 85: 86, 87: 88, 89: 90, 91: 92,  # arms
}


FLIP_ARRAY = np.arange(np.max(np.concatenate(list(FLIP_MAP.items()))) + 1)
for k, v in FLIP_MAP.items():
    FLIP_ARRAY[k] = v
    FLIP_ARRAY[v] = k

# noflip = [1, 2, 5, 6, 7, 8, 12, 14, 15, 18, 20, 24, 25, 27, 28, 33, 34, 35, 36, 40, 41, 42, 43, 48, 49, 50, 51, 52, 54, 55, 56, 62, 63, 64, 65, 66, 67, 71, 72, 73, 76, 77, 78, 79, 82, 83, 85, 87, 88, 89, 90, 91, 92, 95, 98, 100, 103, 107, 110, 111, 112, 113, 117, 118, 119, 120, 121, 124, 125, 126, 129, 130, 131, 133, 137, 139, 140, 141, 145, 147, 148, 150, 151, 153, 159, 161, 162, 164, 166, 167, 168, 170, 172, 174, 175, 176, 179, 182, 183, 186, 188, 189, 190, 193, 196, 197, 198, 200, 201, 202, 203, 205, 206, 207, 209, 211, 212, 213, 217, 218, 219, 222, 224, 226, 227, 228, 230, 233, 237, 238, 240, 241, 243, 244, 245, 246, 247, 248, 249]


def flip(data, flip_array=FLIP_ARRAY, p=1):
    if np.random.random() > p:
        return data
    
#     if data['target'] in noflip:
#         return data

    flip_array_ = np.arange(len(data['x'].T))
    flip_array = flip_array[:len(flip_array_)]
    flip_array_[:len(flip_array)] = flip_array

    data['x'] = - data['x']
    for k in ['x', 'y', 'z']:
        data[k] = data[k].T[flip_array_].T

    return data


FLIP_MAP_HANDS = {
#     41: 74, 42: 75, 43: 76, 44: 77, 45: 78, 46: 79, 47: 80, 48: 81, 49: 82, 50: 83, 51: 84, 52: 85, 53: 86, 54: 87, 55: 88, 56: 89, 57: 90, 58: 91, 59: 92, 60: 93, 61: 94,  # hands
    41: 20, 40: 19, 39: 18, 38: 17, 37: 16, 36: 15, 35: 14, 34: 13, 33: 12, 32: 11, 31: 10, 30: 9, 29: 8, 28: 7, 27: 6, 26: 5, 25: 4, 24: 3, 23: 2, 22: 1, 21: 0,  # hands
    
}

FLIP_ARRAY_HANDS = np.arange(np.max(np.concatenate(list(FLIP_MAP_HANDS.items()))) + 1)
for k, v in FLIP_MAP_HANDS.items():
    FLIP_ARRAY_HANDS[k] = v
    FLIP_ARRAY_HANDS[v] = k

def add_missing_hand(data, flip_array=FLIP_ARRAY_HANDS, p=1.):
    if np.random.random() > p:
        return data
    
    flip_array_ = np.arange(len(data['x'].T))
    flip_array = flip_array[:len(flip_array_)]
    flip_array_[:len(flip_array)] = flip_array

    flipped = {}
    for k in ['x', 'y', 'z']:
        flipped[k] = data[k].T[flip_array_].T
    flipped['x'] = - flipped['x']
    
    shift = data['x'][:, -1].unsqueeze(-1)  # x_nose
    flipped['x'] += 2 * shift
        
    fts = torch.abs(data['x'])
    embed = data['type'][0]

    left_hand = fts.T[embed == 5].T
    right_hand = fts.T[embed == 10].T
    
    missing_left = (left_hand.sum(-1) == 0)
    missing_right = (right_hand.sum(-1) == 0)
    
    for k in ['x', 'y', 'z']:
        data[k][missing_left] = data[k][missing_left] * (embed != 2).unsqueeze(0) + flipped[k][missing_left] * (embed == 2).unsqueeze(0)
        data[k][missing_right] = data[k][missing_right] * (embed != 1).unsqueeze(0) + flipped[k][missing_right] * (embed == 1).unsqueeze(0)

    return data


def normalize(data):
    for k in ['x', 'y', 'z']:
        x = data[k].flatten()
        x = x[x != 0]
        mean = x.mean()
        std = x.std(unbiased=False)

        data[k] = torch.where(
            data[k] != 0,
            (data[k] - mean) / (std + 1e-6),
            0,
        )
    return data


def normalize_face(data):
    embed = data['type'][0]
    for k in ['x', 'y', 'z']:
#         fts = 
        face = data[k].T[embed == 11].T
        
        min_ = face.min(-1)[0].unsqueeze(-1)
        max_= face.max(-1)[0].unsqueeze(-1)

        data[k] = torch.where(
            data[k] != 0,
            (data[k] - min_) / (max_ - min_ + 1e-6),
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


def interpolate(data, p=0.5):
    if np.random.random() > p:
        return data
    
    for k in ['x', 'y', 'z']:
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


def resize(data, p=0.5, max_size=50):
    if np.random.random() > p:
        return data

    try:
        sz = data['x'].size(0)
        size = np.random.randint((sz + 5) // 2, (max_size + sz) // 2)
    except:
        return data
    
    if size <= 2 or size == sz or sz == 1:
        return data

    data = interpolate(data, p=1)

    for k in ["x", "y", "z"]:
        data[k] = torch.nn.functional.interpolate(
            data[k].T.unsqueeze(0), size, mode="linear"
        )[0].T

    data['type'] = data['type'][:1].repeat(size, 1)
    data['landmark'] = data['landmark'][:1].repeat(size, 1)

    return data


def augment(data, aug_strength=3):
    if aug_strength == 3:
        # + mixface
        data = interpolate(data, p=0.5)
        data = flip(data, p=0.5)
        data = rotate(data, p=0.5)
        data = scale(data, p=0.25)

    if aug_strength == 2:
        data = interpolate(data, p=0.5)
        data = flip(data, p=0.5)
        data = rotate(data, p=0.5)
        data = scale(data, p=0.25)

    elif aug_strength == 1:
#         data = interpolate(data, p=0.5)
        data = flip(data, p=0.5)
        data = rotate(data, max_angle=1/12, p=0.25)
#         data = scale(data, p=0.25)

    return data

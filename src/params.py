NUM_WORKERS = 8

DATA_PATH = "../input/"
LOG_PATH = "../logs/"
OUT_PATH = "../output/"

CLASSES = ["cancer"]
NUM_CLASSES = 250

DEVICE = "cuda"

NEPTUNE_PROJECT = "KagglingTheo/Isolated-Sign-Language-Recognition"

FACE_LANDMARKS = {
    "silhouette": [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ][::2],
    "lipsUpperOuter": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291],
    "lipsLowerOuter": [146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    "lipsUpperInner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308],
    "lipsLowerInner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308],
    "rightEyeUpper0": [246, 161, 160, 159, 158, 157, 173][::2],
    "rightEyeLower0": [33, 7, 163, 144, 145, 153, 154, 155, 133][::2],
    #     "rightEyeUpper1": [247, 30, 29, 27, 28, 56, 190],
    #     "rightEyeLower1": [130, 25, 110, 24, 23, 22, 26, 112, 243],
    #     "rightEyeUpper2": [113, 225, 224, 223, 222, 221, 189],
    #     "rightEyeLower2": [226, 31, 228, 229, 230, 231, 232, 233, 244],
    #     "rightEyeLower3": [143, 111, 117, 118, 119, 120, 121, 128, 245],
    "rightEyebrowUpper": [156, 70, 63, 105, 66, 107, 55, 193][::2],
    #     "rightEyebrowLower": [35, 124, 46, 53, 52, 65],
    #     "rightEyeIris": [473, 474, 475, 476, 477],
    "leftEyeUpper0": [466, 388, 387, 386, 385, 384, 398][::2],
    "leftEyeLower0": [263, 249, 390, 373, 374, 380, 381, 382, 362][::2],
    #     "leftEyeUpper1": [467, 260, 259, 257, 258, 286, 414],
    #     "leftEyeLower1": [359, 255, 339, 254, 253, 252, 256, 341, 463],
    #     "leftEyeUpper2": [342, 445, 444, 443, 442, 441, 413],
    #     "leftEyeLower2": [446, 261, 448, 449, 450, 451, 452, 453, 464],
    #     "leftEyeLower3": [372, 340, 346, 347, 348, 349, 350, 357, 465],
    "leftEyebrowUpper": [383, 300, 293, 334, 296, 336, 285, 417][::2],
    #     "leftEyebrowLower": [265, 353, 276, 283, 282, 295],
    #     "leftEyeIris": [468, 469, 470, 471, 472],
    "midwayBetweenEyes": [168],
    "nose": [1, 2, 98, 327],
    "rightCheek": [205],
    "leftCheek": [425],
}

FACE_LANDMARKS = {
    "silhouette": [
        10,
        297,
        284,
        389,
        454,
        361,
        397,
        379,
        400,
        152,
        176,
        150,
        172,
        132,
        234,
        162,
        54,
        67,
    ],
    "lips": [
        61,
        185,
        40,
        39,
        37,
        0,
        267,
        269,
        270,
        409,
        146,
        91,
        181,
        84,
        17,
        314,
        405,
        321,
        375,
        291,
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        308,
        78,
        95,
        88,
        178,
        87,
        14,
        317,
        402,
        318,
        324,
        308,
    ][::2],
    "right_eye": [246, 160, 158, 173, 33, 163, 145, 154, 133],
    "right_eyebrow": [156, 63, 66, 55],
    "left_eye": [466, 387, 385, 398, 263, 390, 374, 381, 362],
    "left_eyebrow": [383, 293, 296, 285],
    "nose": [1, 2, 98, 327, 168],
    "cheeks": [205, 425],
}


POSE_LANDMARKS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
]

ARM_LANDMARKS = [22, 16, 20, 18, 14, 12, 11, 13, 15, 17, 19, 21]


TYPE_MAPPING = {
    "arm": 1,
    "cheeks": 2,
    "left_eye": 3,
    "left_eyebrow": 4,
    "left_hand": 5,
    "lips": 6,
    "nose": 7,
    "right_eye": 8,
    "right_eyebrow": 9,
    "right_hand": 10,
    "silhouette": 11,
}


GRAPH = [  # torch_3
    [4, 6, 0, 26, 24, 35, 40, 33, 36, 34, 37, 15, 18, 14, 17, 13, 21, 16, 4],  # head
    #     [4, 98, 97, 99, 95, 96, 24],  # eyes, nose
    #     [20, 99, 39],  # cheeks
    [5, 3, 2, 22, 23, 25, 32, 29, 9, 12, 5],  # outter lips
    [7, 19, 8, 1, 28, 38, 27, 31, 30, 10, 11, 7],  # inner lips
    #     [17, 5, 7], [27, 25, 36],
    #     [13, 20, 2], [22, 39, 33],
    # RIGHT HAND
    [61, 60, 59, 58],
    [57, 56, 55, 54],
    [53, 52, 51, 50],
    [49, 48, 47, 46],
    [45, 44, 43, 42, 41],
    [58, 54, 50, 46, 41, 58],
    # LEFT HAND
    [94, 93, 92, 91],
    [90, 89, 88, 87],
    [86, 85, 84, 83],
    [82, 81, 80, 79],
    [78, 77, 76, 75, 74],
    [91, 87, 83, 79, 74, 91],
    # Pose
    #     [66, 64, 62, 15, 63, 65, 67],
    #     [67, 69, 71, 73, 67, 71],
    #     [66, 68, 70, 72, 66, 70],
]


GRAPH = [
    [42, 53, 52, 56, 59, 54, 57, 55, 58, 47],  # left head
    [42, 44, 43, 48, 51, 45, 49, 46, 50, 47],  # right head
    [63, 62, 61, 71, 72, 73, 79, 76, 66, 69, 63],  # outter lips
    [64, 70, 65, 60, 75, 80, 74, 78, 77, 67, 68, 64],  # inner lips
    # RIGHT HAND
    [41, 40, 39, 38],
    [37, 36, 35, 34],
    [33, 32, 31, 30],
    [29, 28, 27, 26],
    [25, 24, 23, 22, 21],
    [38, 34, 30, 26, 21, 38],
    # LEFT HAND
    [20, 19, 18, 17],
    [16, 15, 14, 13],
    [12, 11, 10, 9],
    [8, 7, 6, 5],
    [4, 3, 2, 1, 0],
    [17, 13, 9, 5, 0, 17],
    # Arms
    [82, 81, 83, 85, 87, 89, 91, 85],
    [82, 84, 86, 88, 90, 92, 86],
]

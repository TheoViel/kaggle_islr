import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from params import FACE_LANDMARKS, ARM_LANDMARKS


def get_hand_points(hand):
    x = [
        [
            hand.iloc[0].x,
            hand.iloc[1].x,
            hand.iloc[2].x,
            hand.iloc[3].x,
            hand.iloc[4].x,
        ],  # Thumb
        [hand.iloc[5].x, hand.iloc[6].x, hand.iloc[7].x, hand.iloc[8].x],  # Index
        [hand.iloc[9].x, hand.iloc[10].x, hand.iloc[11].x, hand.iloc[12].x],
        [hand.iloc[13].x, hand.iloc[14].x, hand.iloc[15].x, hand.iloc[16].x],
        [hand.iloc[17].x, hand.iloc[18].x, hand.iloc[19].x, hand.iloc[20].x],
        [
            hand.iloc[0].x,
            hand.iloc[5].x,
            hand.iloc[9].x,
            hand.iloc[13].x,
            hand.iloc[17].x,
            hand.iloc[0].x,
        ],
    ]

    y = [
        [
            hand.iloc[0].y,
            hand.iloc[1].y,
            hand.iloc[2].y,
            hand.iloc[3].y,
            hand.iloc[4].y,
        ],  # Thumb
        [hand.iloc[5].y, hand.iloc[6].y, hand.iloc[7].y, hand.iloc[8].y],  # Index
        [hand.iloc[9].y, hand.iloc[10].y, hand.iloc[11].y, hand.iloc[12].y],
        [hand.iloc[13].y, hand.iloc[14].y, hand.iloc[15].y, hand.iloc[16].y],
        [hand.iloc[17].y, hand.iloc[18].y, hand.iloc[19].y, hand.iloc[20].y],
        [
            hand.iloc[0].y,
            hand.iloc[5].y,
            hand.iloc[9].y,
            hand.iloc[13].y,
            hand.iloc[17].y,
            hand.iloc[0].y,
        ],
    ]
    return x, y


def get_pose_points(pose):
    x = [
#         [
#             pose.iloc[8].x,
#             pose.iloc[6].x,
#             pose.iloc[5].x,
#             pose.iloc[4].x,
#             pose.iloc[0].x,
#             pose.iloc[1].x,
#             pose.iloc[2].x,
#             pose.iloc[3].x,
#             pose.iloc[7].x,
#         ],
#         [pose.iloc[10].x, pose.iloc[9].x],
        [
            pose.iloc[22].x,
            pose.iloc[16].x,
            pose.iloc[20].x,
            pose.iloc[18].x,
            pose.iloc[16].x,
            pose.iloc[14].x,
            pose.iloc[12].x,
            pose.iloc[11].x,
            pose.iloc[13].x,
            pose.iloc[15].x,
            pose.iloc[17].x,
            pose.iloc[19].x,
            pose.iloc[15].x,
            pose.iloc[21].x,
        ],
        [
            pose.iloc[12].x,
            pose.iloc[24].x,
            pose.iloc[26].x,
            pose.iloc[28].x,
            pose.iloc[30].x,
            pose.iloc[32].x,
            pose.iloc[28].x,
        ],
        [
            pose.iloc[11].x,
            pose.iloc[23].x,
            pose.iloc[25].x,
            pose.iloc[27].x,
            pose.iloc[29].x,
            pose.iloc[31].x,
            pose.iloc[27].x,
        ],
#         [pose.iloc[24].x, pose.iloc[23].x],
    ]

    y = [
#         [
#             pose.iloc[8].y,
#             pose.iloc[6].y,
#             pose.iloc[5].y,
#             pose.iloc[4].y,
#             pose.iloc[0].y,
#             pose.iloc[1].y,
#             pose.iloc[2].y,
#             pose.iloc[3].y,
#             pose.iloc[7].y,
#         ],
#         [pose.iloc[10].y, pose.iloc[9].y],
        [
            pose.iloc[22].y,
            pose.iloc[16].y,
            pose.iloc[20].y,
            pose.iloc[18].y,
            pose.iloc[16].y,
            pose.iloc[14].y,
            pose.iloc[12].y,
            pose.iloc[11].y,
            pose.iloc[13].y,
            pose.iloc[15].y,
            pose.iloc[17].y,
            pose.iloc[19].y,
            pose.iloc[15].y,
            pose.iloc[21].y,
        ],
        [
            pose.iloc[12].y,
            pose.iloc[24].y,
            pose.iloc[26].y,
            pose.iloc[28].y,
            pose.iloc[30].y,
            pose.iloc[32].y,
            pose.iloc[28].y,
        ],
        [
            pose.iloc[11].y,
            pose.iloc[23].y,
            pose.iloc[25].y,
            pose.iloc[27].y,
            pose.iloc[29].y,
            pose.iloc[31].y,
            pose.iloc[27].y,
        ],
#         [pose.iloc[24].y, pose.iloc[23].y],
    ]
    return x, y


def plot_frame(f, sign, ax, title=''):
    xmin = sign.x.min() - 0.1
    xmax = sign.x.max() + 0.1
    ymin = sign.y.min() - 0.1
    ymax = sign.y.max() + 0.1

    frame = sign[sign.frame == f]
    left = frame[frame.type == "left_hand"]
    right = frame[frame.type == "right_hand"]
    pose = frame[frame.type == "pose"]
    
    face = frame[frame.type == "face"]


    lx, ly = get_hand_points(left)
    rx, ry = get_hand_points(right)
    px, py = get_pose_points(pose)

    ax.clear()
    
    for k in FACE_LANDMARKS.keys():
        ids = FACE_LANDMARKS[k]
        face_ = face[face.landmark_index.isin(ids)][["x", "y"]].values
        ax.plot(face_[:, 0], face_[:, 1], ".", label=k)

    body = pose[pose.landmark_index.isin(ARM_LANDMARKS)]
    body = body[["x", "y"]].values
    ax.plot(body[:, 0], body[:, 1], ".", label="body")

    for i in range(len(lx)):
        ax.plot(lx[i], ly[i])
    for i in range(len(rx)):
        ax.plot(rx[i], ry[i])
    for i in range(len(px)):
        ax.plot(px[i], py[i])

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    
    if title:
        plt.title(title)
        
    plt.legend()

    plt.axis(False)
    plt.tight_layout()
    

def animate(sign, label):
    matplotlib.use('Agg')

    fig, ax = plt.subplots()
    l, = ax.plot([], [])
    animation = FuncAnimation(fig, func=lambda x: plot_frame(x, sign, ax, title=label), frames=sign.frame.unique())

    return HTML(animation.to_html5_video())

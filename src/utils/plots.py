import matplotlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from itertools import product
from IPython.display import HTML
from sklearn.metrics import confusion_matrix
from matplotlib.animation import FuncAnimation

from params import FACE_LANDMARKS, ARM_LANDMARKS, GRAPH


def plot_confusion_matrix(
    y_pred,
    y_true,
    cm=None,
    normalize="true",
    display_labels=None,
    cmap="viridis",
):
    """
    Computes and plots a confusion matrix.
    Args:
        y_pred (numpy array): Predictions.
        y_true (numpy array): Truths.
        normalize (bool or None, optional): Whether to normalize the matrix. Defaults to None.
        display_labels (list of strings or None, optional): Axis labels. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "viridis".
    """
    if cm is None:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
#     cm = cm[::-1, :]

    # Display colormap
    n_classes = cm.shape[0]
    im_ = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    # Display values
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    thresh = (cm.max() + cm.min()) / 2.0
    for i in tqdm(range(n_classes)):
        for j in range(n_classes):
            if cm[i, j] > 0.1:
                color = cmap_max if cm[i, j] < thresh else cmap_min
                text = f"{cm[i, j]:.0f}" if normalize is None else f"{cm[i, j]:.1f}"
                plt.text(j, i, text, ha="center", va="center", color=color)

    # Display legend
    plt.xlim(-0.5, n_classes - 0.5)
    plt.ylim(-0.5, n_classes - 0.5)
    if display_labels is not None:
        plt.xticks(np.arange(n_classes), display_labels)
        plt.yticks(np.arange(n_classes), display_labels)

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)

    
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


def plot_sample(data, n_frames=4, figsize=(10, 10)):
    frames = np.linspace(0, data['x'].shape[0], n_frames, dtype=int, endpoint=False)
    plt.figure(figsize=figsize)
    
    cols = np.array([[0, 0, 0, 0]] + [list(c) + [1] for c in sns.color_palette(n_colors=11)])
    
    for i, frame in enumerate(frames):
        plt.subplot(int(np.sqrt(n_frames)), int(n_frames / int(np.sqrt(n_frames))), i + 1)
        plt.scatter(data['x'][frame], - data['y'][frame], s=4, c=cols[data['type'][frame].numpy().astype(int)])
        plt.title(f"Frame {frame}")
#         plt.grid()
        plt.axis(True)
    plt.show()

    
    
def plot_sample_with_edges(data, n_frames=4, figsize=(10, 10), show_text=False, graph=GRAPH):
    frames = np.linspace(0, data['x'].shape[0], n_frames, dtype=int, endpoint=False)

    plt.figure(figsize=figsize)
    
    cols = np.array([[0, 0, 0, 0]] + [list(c) + [1] for c in sns.color_palette(n_colors=11)])
    
    for i, frame in enumerate(frames):

        plt.subplot(int(np.sqrt(n_frames)), int(n_frames / int(np.sqrt(n_frames))), i + 1)
        plt.scatter(data['x'][frame], - data['y'][frame], s=4, c=cols[data['type'][frame].numpy().astype(int)])
        
        if show_text:
            for i in range(len(data['x'][frame])):
    #             if i not in np.concatenate(GRAPH):
                plt.text(data['x'][frame][i], - data['y'][frame][i], str(i), size=6)
        
        plt.title(f"Frame {frame}")
        plt.axis(True)
    
        for g in graph:
            for i in range(len(g) - 1):
                a = g[i]
                b = g[i + 1]
                plt.plot([data['x'][frame][a], data['x'][frame][b]], [- data['y'][frame][a], - data['y'][frame][b]], c="k", linewidth=0.5)

    plt.show()

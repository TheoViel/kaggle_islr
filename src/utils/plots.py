import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from params import GRAPH


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
        cm (numpy array or None, optional): Precomputed onfusion matrix. Defaults to None.
        normalize (bool or None, optional): Whether to normalize the matrix. Defaults to None.
        display_labels (list of strings or None, optional): Axis labels. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "viridis".
    """
    if cm is None:
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)

    # Display colormap
    n_classes = cm.shape[0]
    im_ = plt.imshow(cm, interpolation="nearest", cmap=cmap)

    # Display values
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(256)
    thresh = (cm.max() + cm.min()) / 2.0
    for i in range(n_classes):
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


def plot_sample(data, n_frames=4, figsize=(10, 10)):
    """
    Plots a sample of data.

    Args:
        data (dict): Data dictionary containing "x", "y", and "type" arrays.
        n_frames (int, optional): Number of frames to plot. Defaults to 4.
        figsize (tuple, optional): Figure size. Defaults to (10, 10).
    """
    frames = np.linspace(0, data["x"].shape[0], n_frames, dtype=int, endpoint=False)
    plt.figure(figsize=figsize)

    cols = np.array(
        [[0, 0, 0, 0]] + [list(c) + [1] for c in sns.color_palette(n_colors=11)]
    )

    for i, frame in enumerate(frames):
        plt.subplot(
            int(np.sqrt(n_frames)), int(n_frames / int(np.sqrt(n_frames))), i + 1
        )
        plt.scatter(
            data["x"][frame],
            -data["y"][frame],
            s=4,
            c=cols[data["type"][frame].numpy().astype(int)],
        )
        plt.title(f"Frame {frame}")
        #         plt.grid()
        plt.axis(False)
    plt.show()


def plot_sample_with_edges(
    data, n_frames=4, figsize=(10, 10), show_text=False, graph=GRAPH
):
    """
    Plots a sample of data with edges connecting points.

    Args:
        data (dict): Data dictionary containing "x", "y", and "type" arrays.
        n_frames (int, optional): Number of frames to plot. Defaults to 4.
        figsize (tuple, optional): Figure size. Defaults to (10, 10).
        show_text (bool, optional): Whether to display text labels for points. Defaults to False.
        graph (list, optional): List of graphs specifying the edges to connect. Defaults to GRAPH.
    """
    frames = np.linspace(0, data["x"].shape[0], n_frames, dtype=int, endpoint=False)

    plt.figure(figsize=figsize)

    cols = np.array(
        [[0, 0, 0, 0]] + [list(c) + [1] for c in sns.color_palette(n_colors=11)]
    )

    for i, frame in enumerate(frames):
        plt.subplot(
            int(np.sqrt(n_frames)), int(n_frames / int(np.sqrt(n_frames))), i + 1
        )
        plt.scatter(
            data["x"][frame],
            -data["y"][frame],
            s=4,
            c=cols[data["type"][frame].numpy().astype(int)],
        )

        if show_text:
            for i in range(len(data["x"][frame])):
                #             if i not in np.concatenate(GRAPH):
                plt.text(data["x"][frame][i], -data["y"][frame][i], str(i), size=6)

        plt.title(f"Frame {frame}")
        plt.axis(True)

        for g in graph:
            for i in range(len(g) - 1):
                a = g[i]
                b = g[i + 1]
                plt.plot(
                    [data["x"][frame][a], data["x"][frame][b]],
                    [-data["y"][frame][a], -data["y"][frame][b]],
                    c="k",
                    linewidth=0.5,
                )

    plt.show()

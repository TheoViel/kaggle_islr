import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


ROWS_PER_FRAME = 543  # number of landmarks per frame


def load_relevant_data_subset(pq_path):
    """
    Loads a relevant subset of data from a Parquet file.

    Args:
        pq_path (str): Path to the Parquet file.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing the DataFrame and the data array.
    """
    df = pd.read_parquet(pq_path)
    n_frames = int(len(df) / ROWS_PER_FRAME)
    data = df[['x', 'y', 'z']].values.reshape(n_frames, ROWS_PER_FRAME, 3)
    return df, data.astype(np.float32)


KEPT_LANDMARKS = [
    [468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488],  # left hand  # noqa
    [522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542],  # right hand  # noqa
    [10, 54, 67, 132, 150, 152, 162, 172, 176, 234, 284, 297, 361, 379, 389, 397, 400, 454],  # silhouette  # noqa
    [13, 37, 40, 61, 78, 81, 84, 87, 88, 91, 191, 267, 270, 291, 308, 311, 314, 317, 318, 321, 415],  # lips  # noqa
    [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511],  # arms
    [205, 425],  # cheeks
]
MAPPING = [i + 1 for i in range(len(KEPT_LANDMARKS))]

TO_AVG = [
    [466, 387, 385, 398, 263, 390, 374, 381, 362],  # left_eye
    [246, 160, 158, 173, 33, 163, 145, 154, 133],
    [383, 293, 296, 285],  # left_eyebrow
    [156, 63, 66, 55],  # right_eyebrow
    [1, 2, 98, 327, 168],  # nose
]


class PreprocessingTF(keras.Model):
    """
    Preprocessing module for data preparation, rewritten in tensorflow.

    Attributes:
        type_embed (tf.Tensor): Type embeddings.
        landmark_embed (tf.Tensor): Landmark embeddings.
        ids (tf.Tensor): Landmark IDs.
        to_avg (List[tf.Tensor]): IDs of landmarks to average.
        hands (tf.Tensor): IDs of hand landmarks.
        frames (tf.Tensor): Frame numbers.
        max_len (tf.Tensor): Maximum length.
        model_max_len (tf.Tensor): Maximum length for the model.

    Methods:
        __init__(self, type_embed, max_len=50, model_max_len=50): Constructor.
        filter_sign(self, x): Filters sign data by removing frames with missing
            hand landmarks, and striding the sequence to a small enough dimension.
        call(self, x): Preprocesses the input sign data.
    """
    def __init__(self, type_embed, max_len=50, model_max_len=50):
        """
        Constructor.

        Args:
            type_embed (numpy array): Type embeddings.
            max_len (int, optional): Maximum length. Defaults to 50.
            model_max_len (int, optional): Model maximum length. Defaults to 50.
        """
        super().__init__()

        self.type_embed = tf.convert_to_tensor(type_embed[None, :].astype(np.float32))
        self.type_embed = tf.repeat(self.type_embed, 1000, axis=0)

        self.landmark_embed = tf.range(100, dtype=tf.float32)[tf.newaxis, :] + 1
        self.landmark_embed = tf.repeat(self.landmark_embed, 100, axis=0)

        self.ids = tf.convert_to_tensor(np.concatenate(KEPT_LANDMARKS))

        self.to_avg = [tf.convert_to_tensor(avg) for avg in TO_AVG]

        self.hands = tf.convert_to_tensor([
            468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484,
            485, 486, 487, 488, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534,
            535, 536, 537, 538, 539, 540, 541, 542
        ])

        self.frames = tf.range(1000) + 1

        self.max_len = tf.constant(max_len, dtype=tf.int32)
        self.model_max_len = tf.constant(model_max_len, dtype=tf.int32)

    def filter_sign(self, x):
        """
        Filters sign data by removing frames with missing hand landmarks,
        and striding the sequence to a small enough dimension.

        Args:
            x (tf.Tensor): Sign data.

        Returns:
            tf.Tensor: Filtered sign data.
        """
        hands = tf.gather(x, self.hands, axis=1)[:, :, 0]
        nan_prop = tf.reduce_mean(tf.cast(tf.math.is_nan(hands), dtype=tf.float32), axis=-1)
        nan_mask = nan_prop < 1
        x = tf.boolean_mask(x, nan_mask)

        length = tf.reduce_max(self.frames[:tf.shape(x)[0]])
        sz = tf.reduce_max(tf.stack([length, self.max_len], axis=0))

        divisor = tf.where(sz - self.max_len > 0, sz // self.max_len + 1, 1)
        ids = tf.math.floormod(self.frames[:tf.shape(x)[0]], divisor) == 0
        x = tf.boolean_mask(x, ids)

        return x

    def call(self, x):
        """
        Forward pass of the preprocessing module.

        Args:
            x (tf.Tensor): Input sign data.

        Returns:
            tf.Tensor: Processed sign data.
        """
        x = self.filter_sign(x)
        n_frames = tf.reduce_max(self.frames[:tf.shape(x)[0]])

        avg_ids = []
        for ids in self.to_avg:
            avg_id = tf.math.reduce_mean(tf.gather(x, ids, axis=1), axis=1, keepdims=True)
            avg_ids.append(avg_id)

        x = tf.concat([tf.gather(x, self.ids, axis=1)] + avg_ids, axis=1)

        type_embed = self.type_embed[:n_frames]
        landmark_embed = self.landmark_embed[:n_frames]

        # Normalize & fill nans
        nonan = tf.boolean_mask(x, tf.math.logical_not(tf.math.is_nan(x)))
        nonan = tf.reshape(nonan, (-1, 3))
        x = x - tf.math.reduce_mean(nonan, axis=0)[tf.newaxis, tf.newaxis, :]
        x = x / tf.math.reduce_std(nonan, axis=0)[tf.newaxis, tf.newaxis, :]
        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

        # Concat
        x = tf.concat([
            tf.expand_dims(type_embed, -1), x, tf.expand_dims(landmark_embed, -1)
        ], -1)
        x = tf.transpose(x, [0, 2, 1])

        x = x[:self.model_max_len]

        return x

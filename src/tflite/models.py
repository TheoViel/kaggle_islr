import torch
import nobuco
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LayerNorm
from transformers import AutoConfig
from tflite.deberta import DebertaV2Encoder


class DebertaV2Output(nn.Module):
    """
    Modified DebertaV2Output Layer. We changed the position of the skip connection,
    to allow for output_size != intermediate_size.

    Attributes:
        dense (Linear): The linear transformation layer.
        LayerNorm (LayerNorm): The layer normalization layer.
        config (DebertaV2Config): The model configuration class instance.

    Methods:
        __init__(self, config): Initializes a DebertaV2Output instance with the specified config.
        forward(self, hidden_states, input_tensor): Performs the forward pass.
    """
    def __init__(self, config):
        """
        Constructor.

        Args:
            config (DebertaV2Config): The model configuration class instance.
        """
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.output_size)
        self.LayerNorm = LayerNorm(config.output_size, config.layer_norm_eps)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        """
        Performs the forward pass.

        Args:
            hidden_states (Tensor): The hidden states from the previous layer.
            input_tensor (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        hidden_states = self.dense(hidden_states + input_tensor)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


def compute_ids(transpose=True, max_len=50, position_buckets=50):
    """
    Precomputes the positional IDs for the transformer model.

    Args:
        transpose (bool, optional): Whether to transpose the positional IDs. Defaults to True.
        max_len (int, optional): Maximum length of the sequence. Defaults to 50.
        position_buckets (int, optional): Number of position buckets. Defaults to 50.

    Returns:
        torch.Tensor: Computed positional IDs.
    """
    if transpose:
        ids = (
            np.arange(max_len)[None]
            - np.arange(max_len)[:, None]
            + position_buckets
        )
    else:
        ids = (
            np.arange(max_len)[:, None]
            - np.arange(max_len)[None]
            + position_buckets
        )

    ids += np.arange(0, max_len)[:, None] * position_buckets * 2
    ids = ids[None]
    ids = torch.from_numpy(ids)
    return ids


class Model(nn.Module):
    """
    Rewrites the SignMLPBert3 class to be optimized for TFLite inference.

    Attributes:
        num_classes (int): The number of classes for the main task.
        num_classes_aux (int): The number of classes for auxiliary tasks.
        transfo_heads (int): The number of attention heads in the transformer.
        transfo_layers (int): The number of transformer layers.
        max_len (int): The maximum length of input sequences.

        type_embed (nn.Embedding): Embedding layer for the type input.
        landmark_embed (nn.Embedding): Embedding layer for the landmark input.
        type_norm (nn.LayerNorm): Layer normalization for the type embeddings.
        landmark_norm (nn.LayerNorm): Layer normalization for the landmark embeddings.
        pos_cnn (nn.Sequential): CNN layers for positional encoding.
        pos_dense (nn.Linear): Dense layer for positional encoding.
        dense (nn.Linear): Dense layer for combining input features.
        left_hand_mlp (nn.Sequential): MLP layers for left hand features.
        right_hand_mlp (nn.Sequential): MLP layers for right hand features.
        lips_mlp (nn.Sequential): MLP layers for lips features.
        face_mlp (nn.Sequential): MLP layers for face features.
        full_mlp (nn.Sequential): MLP layers for all features.
        landmark_mlp (nn.Sequential): MLP layers for landmark features.
        frame_transformer_1 (DebertaV2Encoder): First DeBERTa transformer layer.
        frame_transformer_2 (DebertaV2Encoder): Second DeBERTa transformer layer.
        frame_transformer_3 (DebertaV2Encoder): Third DeBERTa transformer layer.
        logits (nn.Linear): Linear layer for main task classification.

        offset (torch.Tensor): Precomputed offsets for relative position embeddings
        ids (torch.Tensor): Precomputed ids for relative position embeddings
        ids_t (torch.Tensor): Precomputed transposed ids for relative position embeddings

    Methods:
        __init__(self, embed_dim, transfo_dim, dense_dim, transfo_heads, transfo_layers, drop_rate,
                 num_classes,num_classes_aux, n_landmarks, max_len): Constructor
        forward(self, x, perm, coefs): Performs the forward pass.
    """
    def __init__(
        self,
        embed_dim=256,
        dense_dim=384,
        transfo_dim=768,
        transfo_layers=3,
        transfo_heads=1,
        num_classes=250,
        drop_rate=0,
        n_landmarks=100,
        max_len=50,
    ):
        """
        Constructor.

        Args:
            embed_dim (int, optional): The embedding dimension. Defaults to 256.
            dense_dim (int, optional): The dense layer dimension. Defaults to 512.
            transfo_dim (int, optional): The transformer layer dimension. Defaults to 768.
            transfo_layers (int, optional): The number of transformer layers. Defaults to 4.
            transfo_heads (int, optional): Number of attention heads. Defaults to 8.
            num_classes (int, optional): The number of classes for the main task. Defaults to 250.
            drop_rate (float, optional): The dropout rate. Defaults to 0.
            n_landmarks (int, optional): The number of landmarks. Defaults to 100.
            max_len (int, optional): The maximum length of input sequences. Defaults to 40.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_aux = 0
        self.transfo_heads = transfo_heads

        self.type_embed = nn.Embedding(9, embed_dim, padding_idx=0)
        self.landmark_embed = nn.Embedding(101, embed_dim, padding_idx=0)
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)

        self.pos_cnn = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=5, padding=2, bias=False),
            nn.Conv1d(8, 16, kernel_size=5, padding=2, bias=False),
        )
        self.pos_dense = nn.Linear(19, embed_dim)

        self.dense = nn.Linear(3 * embed_dim, embed_dim)

        self.left_hand_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_rate),
            nn.Mish(),
        )

        self.right_hand_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_rate),
            nn.Mish(),
        )

        self.lips_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_rate),
            nn.Mish(),
        )

        self.face_mlp = nn.Sequential(
            nn.Linear(embed_dim * 25, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_rate),
            nn.Mish(),
        )

        self.full_mlp = nn.Sequential(
            nn.Linear(embed_dim * n_landmarks, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_rate),
            nn.Mish(),
        )

        transfo_dim_ = transfo_dim
        if transfo_layers == 3:  # 512, 768, 1024 / 768
            if transfo_dim <= 1024:
                delta = min(256, transfo_dim - 512)
                transfo_dim = 512
            else:  # BIG Models
                delta = (transfo_dim - 1024) // 2
                transfo_dim = 1024
        else:  # 768, 768
            delta = 0
        self.transfo_dim = transfo_dim

        self.landmark_mlp = nn.Sequential(
            nn.Linear(dense_dim * 4, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.Mish(),
        )

        name = "microsoft/deberta-v3-base"

        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim
        config.output_size = transfo_dim
        if transfo_layers >= 2:
            config.output_size = transfo_dim + delta
        config.num_hidden_layers = 1
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate
        config.hidden_dropout_prob = drop_rate
        config.hidden_act = nn.Mish()  # "relu"
        config.max_relative_positions = max_len
        config.position_buckets = max_len
        config.max_len = max_len
        self.max_len = max_len

        self.frame_transformer_1 = DebertaV2Encoder(config)
        self.frame_transformer_1.layer[0].output = DebertaV2Output(config)

        self.frame_transformer_2 = None
        if transfo_layers >= 2:
            config.hidden_size += delta
            config.intermediate_size += delta

            if transfo_layers >= 3 and transfo_dim_ >= 1024:
                config.output_size += delta

            if delta > 0:
                config.attention_probs_dropout_prob *= 2
                config.hidden_dropout_prob *= 2

            self.frame_transformer_2 = DebertaV2Encoder(config)
            self.frame_transformer_2.layer[0].output = DebertaV2Output(config)

        self.frame_transformer_3 = None
        if transfo_layers >= 3:
            if transfo_dim_ >= 1024:
                config.hidden_size += delta
                config.intermediate_size += delta
                config.attention_probs_dropout_prob *= 2
                config.hidden_dropout_prob *= 2
                config.output_size += delta

            config.output_size -= delta
            self.frame_transformer_3 = DebertaV2Encoder(config)
            self.frame_transformer_3.layer[0].output = DebertaV2Output(config)

        self.logits = nn.Linear(config.output_size, num_classes)

        self.ids = torch.zeros(
            (
                config.max_len,
                1,
                config.max_len,
                config.max_len,
            ),
            dtype=torch.int,
        )
        self.ids_t = torch.zeros(
            (
                config.max_len,
                1,
                config.max_len,
                config.max_len,
            ),
            dtype=torch.int,
        )
        for k in range(1, config.max_len + 1):
            self.ids[k - 1, :, :k, :k] = compute_ids(
                transpose=False, max_len=k, position_buckets=max_len
            )
            self.ids_t[k - 1, :, :k, :k] = compute_ids(
                transpose=True, max_len=k, position_buckets=max_len
            )

        self.offset = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        ) * 2 * max_len
        self.offset = self.offset.int()

    def forward(self, x):
        """
        Performs the forward pass.

        Args:
            x (Tensor): The input tensor containing sign language data.
            perm (Tensor, optional): Permutation tensor for Manifold Mixup. Defaults to None.
            coefs (Tensor, optional): Coefficients tensor for Manifold Mixup. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: The main task logits and auxiliary task logits (if applicable).
        """
        x = x.unsqueeze(0)

        bs, sz, n_fts, n_landmarks = nobuco.shape(x)

        x_type = self.type_norm(self.type_embed(x[:, :, 0].long()))
        x_landmark = self.landmark_norm(self.landmark_embed(x[:, :, 4].long()))

        x_pos_ = x[:, :, 1:4].transpose(2, 3).contiguous()
        x_pos_ = F.pad(x_pos_, (0, 0, 0, 0, 0, 2))

        x_pos = x_pos_.transpose(1, 2).transpose(2, 3).contiguous().view(bs * n_landmarks, 3, -1)
        x_pos = self.pos_cnn(x_pos)
        x_pos = x_pos.view(bs, n_landmarks, 16, -1).transpose(2, 3).transpose(1, 2).contiguous()

        x_pos = torch.cat([x_pos_, x_pos], -1)
        x_pos = x_pos[:, :-2]

        x_pos = self.pos_dense(x_pos)

        fts = self.dense(torch.cat([x_type, x_landmark, x_pos], -1))

        n_fts = fts.size(-1)
        embed = x[:, :, 0].contiguous().unsqueeze(1).view(-1).long()

#         left_hand_fts = fts.view(-1, n_fts)[embed == 1].view(-1, 21 * n_fts)
        left_hand_fts = fts.view(-1, n_fts)[embed == 1].view(bs, -1, 21, n_fts)
        left_hand_fts -= left_hand_fts.mean(1).mean(1).unsqueeze(1).unsqueeze(1)
        left_hand_fts = left_hand_fts.view(-1, 21 * n_fts)
        left_hand_fts = self.left_hand_mlp(left_hand_fts)

        right_hand_fts = fts.view(-1, n_fts)[embed == 2].view(bs, -1, 21, n_fts)
        right_hand_fts -= right_hand_fts.mean(1).mean(1).unsqueeze(1).unsqueeze(1)
        right_hand_fts = right_hand_fts.view(-1, 21 * n_fts)
        right_hand_fts = self.right_hand_mlp(right_hand_fts)

        hand_fts = torch.stack([left_hand_fts, right_hand_fts], -1).amax(-1)

        lips_fts = fts.view(-1, n_fts)[embed == 4].view(bs, -1, 21, n_fts)
        lips_fts -= lips_fts.mean(1).mean(1).unsqueeze(1).unsqueeze(1)
        lips_fts = lips_fts.view(-1, 21 * n_fts)
        lips_fts = self.lips_mlp(lips_fts)

        face_fts = fts.view(-1, n_fts)[(embed == 3) | (embed == 6)].view(bs, -1, 25, n_fts)
        face_fts -= face_fts.mean(1).mean(1).unsqueeze(1).unsqueeze(1)
        face_fts = face_fts.view(-1, 25 * n_fts)
        face_fts = self.face_mlp(face_fts)

        fts = fts.view(-1, n_fts * n_landmarks)

        fts = self.full_mlp(fts)

        fts = torch.cat([fts, hand_fts, lips_fts, face_fts], -1)

        fts = self.landmark_mlp(fts)
        fts = fts.view(bs, -1, self.transfo_dim)

        ids_t = self.ids_t[sz - 1, :, :sz, :sz].contiguous()
        ids = self.ids[sz - 1, :, :sz, :sz].contiguous()
        offset = (self.offset.unsqueeze(1).unsqueeze(1) * sz)
        ids = (ids + offset).view(-1)
        ids_t = (ids_t + offset).view(-1)

        fts = self.frame_transformer_1(fts, ids=ids, ids_t=ids_t).last_hidden_state
        if self.frame_transformer_2 is not None:
            fts = self.frame_transformer_2(fts, ids=ids, ids_t=ids_t).last_hidden_state
        if self.frame_transformer_3 is not None:
            fts = self.frame_transformer_3(fts, ids=ids, ids_t=ids_t).last_hidden_state

        fts = fts.mean(1)

        logits = self.logits(fts)

        return logits

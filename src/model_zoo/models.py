import torch
import numpy as np
import torch.nn as nn

from torch.nn import LayerNorm
from torch.masked import masked_tensor
from transformers import AutoConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout

from utils.torch import load_model_weights


def define_model(
    name,
    embed_dim=256,
    dense_dim=512,
    transfo_dim=768,
    transfo_heads=8,
    transfo_layers=4,
    drop_rate=0,
    num_classes=250,
    num_classes_aux=0,
    pretrained_weights="",
    n_landmarks=100,
    max_len=40,
    verbose=1,
):
    """
    Defines and initializes a specific model architecture.

    Args:
        name (str): The name of the model architecture.
        embed_dim (int, optional): The embedding dimension. Defaults to 256.
        dense_dim (int, optional): The dense layer dimension. Defaults to 512.
        transfo_dim (int, optional): The transformer layer dimension. Defaults to 768.
        transfo_heads (int, optional): The number of attention heads in the transformer. Defaults to 8.
        transfo_layers (int, optional): The number of transformer layers. Defaults to 4.
        drop_rate (float, optional): The dropout rate. Defaults to 0.
        num_classes (int, optional): The number of classes for the main task. Defaults to 250.
        num_classes_aux (int, optional): The number of classes for auxiliary tasks. Defaults to 0.
        pretrained_weights (str, optional): Path to pre-trained weights to initialize the model. Defaults to "".
        n_landmarks (int, optional): The number of landmarks. Defaults to 100.
        max_len (int, optional): The maximum length of input sequences. Defaults to 40.
        verbose (int, optional): Verbosity level. Defaults to 1.

    Returns:
        torch.nn.Module: The initialized model.

    Raises:
        NotImplementedError: If the specified model architecture is not implemented.
    """
    if name == "mlp_bert_3" or name == "mlp_bert_4":
        model = SignMLPBert3(
            embed_dim=embed_dim,
            dense_dim=dense_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            transfo_layers=transfo_layers,
            num_classes=num_classes,
            num_classes_aux=num_classes_aux,
            n_landmarks=n_landmarks,
            drop_rate=drop_rate,
            max_len=max_len,
        )
    else:
        raise NotImplementedError

    if pretrained_weights:
        model = load_model_weights(
            model, pretrained_weights, verbose=verbose, strict=False
        )

    model.name = name

    return model


class DebertaV2Output(nn.Module):
    """
    Modified DebertaV2Output Layer. We changed the position of the skip connection,
    to allow for output_size != intermediate_size.

    Attributes:
        dense (Linear): The linear transformation layer.
        LayerNorm (LayerNorm): The layer normalization layer.
        dropout (StableDropout): The dropout layer.
        config (DebertaV2Config): The model configuration class instance.

    Methods:
        __init__(self, config): Initializes a DebertaV2Output instance with the specified configuration.
        forward(self, hidden_states, input_tensor): Performs the forward pass of the DebertaV2Output model.
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
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        """
        Performs the forward pass of the DebertaV2Output model.

        Args:
            hidden_states (Tensor): The hidden states from the previous layer.
            input_tensor (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying the linear transformation, dropout, and layer normalization.
        """
        if self.config.skip_output:
            hidden_states = self.dense(hidden_states + input_tensor)
        else:
            hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SignMLPBert3(nn.Module):
    """
    MLP + Deberta architecture.

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
        frame_transformer_2 (DebertaV2Encoder): Second DeBERTa transformer layer (optional).
        frame_transformer_3 (DebertaV2Encoder): Third DeBERTa transformer layer (optional).
        logits (nn.Linear): Linear layer for main task classification.
        logits_aux (nn.Linear): Linear layer for auxiliary task classification (optional).

    Methods:
        __init__(self, embed_dim, transfo_dim, dense_dim, transfo_heads, transfo_layers, drop_rate, num_classes,
                 num_classes_aux, n_landmarks, max_len): Constructor
        forward(self, x, perm, coefs): Performs the forward pass.
    """
    def __init__(
        self,
        embed_dim=256,
        transfo_dim=768,
        dense_dim=512,
        transfo_heads=1,
        transfo_layers=4,
        drop_rate=0,
        num_classes=250,
        num_classes_aux=0,
        n_landmarks=100,
        max_len=40,
    ):
        """
        Constructor.
        
        Args:
            embed_dim (int, optional): The embedding dimension. Defaults to 256.
            dense_dim (int, optional): The dense layer dimension. Defaults to 512.
            transfo_dim (int, optional): The transformer layer dimension. Defaults to 768.
            transfo_heads (int, optional): The number of attention heads in the transformer. Defaults to 8.
            drop_rate (float, optional): The dropout rate. Defaults to 0.
            transfo_layers (int, optional): The number of transformer layers. Defaults to 4.
            num_classes (int, optional): The number of classes for the main task. Defaults to 250.
            num_classes_aux (int, optional): The number of classes for auxiliary tasks. Defaults to 0.
            n_landmarks (int, optional): The number of landmarks. Defaults to 100.
            max_len (int, optional): The maximum length of input sequences. Defaults to 40.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.transfo_heads = transfo_heads
        self.transfo_layers = transfo_layers
        self.max_len = max_len

        self.type_embed = nn.Embedding(9, embed_dim, padding_idx=0)
        self.landmark_embed = nn.Embedding(n_landmarks + 1, embed_dim, padding_idx=0)
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)

        self.pos_cnn = nn.Sequential(
            nn.Conv1d(3, 8, kernel_size=5, padding=2, bias=False),
            nn.Conv1d(8, 16, kernel_size=5, padding=2, bias=False),
        )
        self.pos_dense = nn.Linear(19, embed_dim)

        self.dense = nn.Linear(3 * embed_dim, embed_dim)

        drop_mlp = drop_rate if dense_dim >= 256 else drop_rate / 2
        self.left_hand_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_mlp),
            nn.Mish(),
        )

        self.right_hand_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_mlp),
            nn.Mish(),
        )

        self.lips_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_mlp),
            nn.Mish(),
        )

        self.face_mlp = nn.Sequential(
            nn.Linear(embed_dim * 25, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_mlp),
            nn.Mish(),
        )

        self.full_mlp = nn.Sequential(
            nn.Linear(embed_dim * n_landmarks, dense_dim),
            nn.BatchNorm1d(dense_dim),
            nn.Dropout(p=drop_mlp),
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
        config.skip_output = True

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
        if num_classes_aux:
            self.logits_aux = nn.Linear(config.output_size, num_classes_aux)
            

    def forward(self, x, perm=None, coefs=None):
        """
        Performs the forward pass.

        Args:
            x (Tensor): The input tensor containing sign language data.
            perm (Tensor, optional): Permutation tensor for Manifold Mixup. Defaults to None.
            coefs (Tensor, optional): Coefficients tensor for Manifold Mixup. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: The main task logits and auxiliary task logits (if applicable).
        """
        bs, n_frames, n_landmarks = x["x"].size()

        x_type = self.type_norm(self.type_embed(x["type"]))
        x_landmark = self.landmark_norm(self.landmark_embed(x["landmark"]))
        x_pos = torch.stack([x["x"], x["y"], x["z"]], -1)

        x_pos = x_pos.transpose(1, 2).transpose(2, 3).contiguous().view(bs * n_landmarks, -1, n_frames)
        x_pos = self.pos_cnn(x_pos)
        x_pos = x_pos.view(bs, n_landmarks, -1, n_frames).transpose(2, 3).transpose(1, 2).contiguous()
        x_pos = torch.cat([torch.stack([x["x"], x["y"], x["z"]], -1), x_pos], -1)

        x_pos = self.pos_dense(x_pos)

        fts = self.dense(torch.cat([x_type, x_landmark, x_pos], -1))

        n_fts = fts.size(-1)
        embed = x["type"][:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)  # this handles padding

        # LEFT HAND FEATURES
        left_hand_fts = fts.view(-1, n_fts)[embed == 1].view(bs, n_frames, -1, n_fts)
        
        mask = x["mask"][:, :, 0][:, :, None, None]
        mask_left_hand = mask.repeat(1, 1, left_hand_fts.shape[2], n_fts)
        m = left_hand_fts * mask_left_hand
        m = m.sum((1, 2)) / mask_left_hand.sum((1, 2))  # masked avg
        left_hand_fts = left_hand_fts - m[:, None, None, :]
        left_hand_fts = left_hand_fts * mask_left_hand
        left_hand_fts = self.left_hand_mlp(left_hand_fts.view(bs * n_frames, -1))

        # RIGHT HAND FEATURES
        right_hand_fts = fts.view(-1, n_fts)[embed == 2].view(bs, n_frames, -1, n_fts)
        
        mask_right_hand = mask.repeat(1, 1, right_hand_fts.shape[2], n_fts)
        m = right_hand_fts * mask_right_hand
        m = m.sum((1, 2)) / mask_right_hand.sum((1, 2))  # masked avg
        right_hand_fts = right_hand_fts - m[:, None, None, :]
        right_hand_fts = right_hand_fts * mask_right_hand
        
        right_hand_fts = self.right_hand_mlp(right_hand_fts.view(bs * n_frames, -1))

        hand_fts = torch.stack([left_hand_fts, right_hand_fts], -1).amax(-1)

        # LIPS FEATURES
        lips_fts = fts.view(-1, n_fts)[embed == 4].view(bs, n_frames, -1, n_fts)
        
        mask_lips = mask.repeat(1, 1, lips_fts.shape[2], n_fts)
        m = lips_fts * mask_lips
        m = m.sum((1, 2)) / mask_lips.sum((1, 2))  # masked avg
        lips_fts = lips_fts - m[:, None, None, :]
        lips_fts = lips_fts * mask_lips
        
        lips_fts = self.lips_mlp(lips_fts.view(bs * n_frames, -1))

        # FACE FEATURES
        face_fts = fts.view(-1, n_fts)[
            torch.isin(embed, torch.tensor([3, 6]).to(fts.device))
        ].view(bs, n_frames, -1, n_fts)

        mask_face = mask.repeat(1, 1, face_fts.shape[2], n_fts)
        m = face_fts * mask_face
        m = m.sum((1, 2)) / mask_face.sum((1, 2))  # masked avg
        face_fts = face_fts - m[:, None, None, :]
        face_fts = face_fts * mask_face

        face_fts = self.face_mlp(face_fts.view(bs * n_frames, -1))

        # ALL FEATURES
        fts = fts.view(bs * n_frames, -1)
        fts = self.full_mlp(fts)

        fts = torch.cat([fts, hand_fts, lips_fts, face_fts], -1)  # dists_fts, hand_edge_fts

        fts = self.landmark_mlp(fts)
        fts = fts.view(bs, n_frames, -1)

        mask = x["mask"][:, :, 0][:, :n_frames]
        mask_avg = mask.unsqueeze(-1)

        fts *= mask.unsqueeze(-1)

        where = np.random.randint(1, self.transfo_layers + 1) if perm is not None else 0  # Manifold Mixup
        coefs = coefs.view(-1, 1, 1)  if coefs is not None else None

#         print(mask_avg.size(), fts.size())
        if where == 1:
            fts = coefs * fts + (1 - coefs) * fts[perm]
#             mask_avg = coefs * mask_avg + (1 - coefs) * mask_avg[perm]
            mask = ((mask + mask[perm]) > 0).float()
            mask_avg = mask.unsqueeze(-1)

        fts = self.frame_transformer_1(fts, mask).last_hidden_state
        
        if where == 2:
            fts = coefs * fts + (1 - coefs) * fts[perm]
#             mask_avg = coefs * mask_avg + (1 - coefs) * mask_avg[perm]
            mask = ((mask + mask[perm]) > 0).float()
            mask_avg = mask.unsqueeze(-1)

        if self.frame_transformer_2 is not None:
            fts = self.frame_transformer_2(fts, mask).last_hidden_state
            
        if where == 3:
            fts = coefs * fts + (1 - coefs) * fts[perm]
#             mask_avg = coefs * mask_avg + (1 - coefs) * mask_avg[perm]
            mask = ((mask + mask[perm]) > 0).float()
            mask_avg = mask.unsqueeze(-1)

        if self.frame_transformer_3 is not None:
            fts = self.frame_transformer_3(fts, mask).last_hidden_state

        fts = fts * mask_avg
        fts = fts.sum(1) / mask_avg.sum(1)  # masked avg

        logits = self.logits(fts)
        logits_aux = self.logits_aux(fts) if self.num_classes_aux else torch.zeros(1)
        return logits, logits_aux

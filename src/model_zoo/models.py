import torch
import numpy as np
import torch.nn as nn

from torch.nn import LayerNorm
from transformers import AutoConfig
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder
from transformers.models.deberta_v2.modeling_deberta_v2 import StableDropout

from model_zoo.resnet import ResNet, Bottleneck
from model_zoo.utils import (
    add_shift,
    get_edge_features,
    # compute_finger_face_distance,
    # compute_hand_features,
)
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
    Builds the architecture.
    TODO

    Args:
        name (str): Model name
        num_classes (int, optional): Number of classes. Defaults to 1.
        pretrained (bool, optional): Whether to load timm pretrained weights.
        verbose (int, optional): Whether to display infos. Defaults to 1.

    Returns:
        torch Module: Model.
    """
    if name == "mlp_bert_3":
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
    elif name == "mlp_bert_skip":
        model = SignMLPBertSkip(
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
    elif name == "mlp_cnn":
        model = SignMLPCNN(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    else:
        raise NotImplementedError

    if pretrained_weights:
        model = load_model_weights(
            model, pretrained_weights, verbose=verbose, strict=False
        )

    model.name = name

    return model


class SignMLPCNN(nn.Module):
    """
    Model with an attention mechanism.
    """

    def __init__(
        self,
        embed_dim=256,
        transfo_dim=768,
        transfo_heads=1,
        num_classes=250,
        n_landmarks=100,
        transfo_layers=4,
        drop_rate=0,
    ):
        """
        Constructor.

        Args:
            encoder (timm model): Encoder.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_aux = 0
        self.transfo_heads = transfo_heads
        self.multi_sample_dropout = False

        self.type_embed = nn.Embedding(12, embed_dim, padding_idx=0)
        self.landmark_embed = nn.Embedding(n_landmarks + 1, embed_dim, padding_idx=0)
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)

        self.pos_dense = nn.Linear(9, embed_dim)
        self.dense = nn.Linear(3 * embed_dim, embed_dim)

        self.left_hand_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

        self.right_hand_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

        self.lips_mlp = nn.Sequential(
            nn.Linear(embed_dim * 21, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

        self.face_mlp = nn.Sequential(
            nn.Linear(embed_dim * 25, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

        self.full_mlp = nn.Sequential(
            nn.Linear(embed_dim * n_landmarks, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

        self.landmark_mlp = nn.Sequential(
            nn.Linear(transfo_dim * 4, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

        self.cnn = ResNet(
            Bottleneck, [1, 1, 1, 1], ft_dim=transfo_dim, n_layers=transfo_layers
        )

        self.logits = nn.Linear(self.cnn.layers[-1][0].conv3.out_channels, num_classes)

    def forward(self, x, return_fts=False):
        """
        Forward function.

        Args:
            x (torch tensor [batch_size x c x h x w]): Input batch.
            return_fts (bool, Optional): Whether to return encoder features.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
            torch tensor [batch_size x num_features]: Encoder features, if return_fts.
        """
        bs, n_frames, n_landmarks = x["x"].size()

        x_type = self.type_norm(self.type_embed(x["type"]))
        x_landmark = self.landmark_norm(self.landmark_embed(x["landmark"]))
        x_pos = torch.stack([x["x"], x["y"], x["z"]], -1)        

        x_pos = add_shift(x_pos)
        x_pos = self.pos_dense(x_pos)

        fts = self.dense(torch.cat([x_type, x_landmark, x_pos], -1))

        n_fts = fts.size(-1)
        embed = x["type"][:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)

        left_hand_fts = fts.view(-1, n_fts)[embed == 5].view(bs, n_frames, -1, n_fts)
        left_hand_fts = self.left_hand_mlp(left_hand_fts.view(bs * n_frames, -1))

        right_hand_fts = fts.view(-1, n_fts)[embed == 10].view(bs, n_frames, -1, n_fts)
        right_hand_fts = self.right_hand_mlp(right_hand_fts.view(bs * n_frames, -1))

        hand_fts = torch.stack([left_hand_fts, right_hand_fts], -1).max(-1).values

        lips_fts = fts.view(-1, n_fts)[embed == 6].view(bs, n_frames, -1, n_fts)
        lips_fts = self.lips_mlp(lips_fts.view(bs * n_frames, -1))

        face_fts = fts.view(-1, n_fts)[
            torch.isin(embed, torch.tensor([11, 2, 3, 4, 8, 9, 7]).to(fts.device))
        ].view(bs, n_frames, -1, n_fts)
        face_fts = self.face_mlp(face_fts.view(bs * n_frames, -1))

        fts = fts.view(-1, n_fts).view(bs, n_frames, -1, n_fts)
        fts = fts.view(bs * n_frames, -1)

        fts = self.full_mlp(fts)

        fts = torch.cat([fts, hand_fts, lips_fts, face_fts], -1)

        fts = self.landmark_mlp(fts)
        fts = fts.view(bs, n_frames, -1).transpose(1, 2)

        fts = self.cnn(fts)

        fts = fts.mean(-1)

        if self.multi_sample_dropout and self.training:
            logits = torch.stack(
                [self.logits(self.dropout(fts)) for _ in range(5)],
                dim=0,
            ).mean(0)
        else:
            logits = self.logits(fts)

        return logits, torch.zeros(1)


class DebertaV2Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.output_size)
        self.LayerNorm = LayerNorm(config.output_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        #         print(hidden_states.size(), input_tensor.size())
        if self.config.skip_output:
            hidden_states = self.dense(hidden_states + input_tensor)
        else:
            hidden_states = self.dense(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class SignMLPBert3(nn.Module):
    """
    Model with an attention mechanism.
    """

    def __init__(
        self,
        embed_dim=256,
        transfo_dim=768,
        dense_dim=512,
        transfo_heads=1,
        n_landmarks=100,
        transfo_layers=4,
        num_classes=250,
        num_classes_aux=0,
        drop_rate=0,
        max_len=40,
    ):
        """
        Constructor.

        Args:
            encoder (timm model): Encoder.
            num_classes (int, optional): Number of classes. Defaults to 1.
            num_classes_aux (int, optional): Number of aux classes. Defaults to 0.
            n_channels (int, optional): Number of image channels. Defaults to 3.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.transfo_heads = transfo_heads
        self.transfo_layers = transfo_layers
        self.multi_sample_dropout = False
        self.use_cnn = True
        self.max_len = max_len

        self.type_embed = nn.Embedding(9, embed_dim, padding_idx=0)
        self.landmark_embed = nn.Embedding(n_landmarks + 1, embed_dim, padding_idx=0)
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)

        if self.use_cnn:
            self.pos_cnn = nn.Sequential(
                nn.Conv1d(3, 8, kernel_size=5, padding=2, bias=False),
                nn.Conv1d(8, 16, kernel_size=5, padding=2, bias=False),
            )
            self.pos_dense = nn.Linear(19, embed_dim)
        else:
            self.pos_dense = nn.Linear(9, embed_dim)

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

#         self.left_hand_edge_dense = nn.Linear(3, embed_dim)
#         self.left_hand_edge_mlp = nn.Sequential(
#             nn.Linear(embed_dim * 15, dense_dim),
#             nn.BatchNorm1d(dense_dim),
#             nn.Dropout(p=drop_mlp),
#             nn.Mish(),
#         )

#         self.right_hand_edge_dense = nn.Linear(3, embed_dim)
#         self.right_hand_edge_mlp = nn.Sequential(
#             nn.Linear(embed_dim * 15, dense_dim),
#             nn.BatchNorm1d(dense_dim),
#             nn.Dropout(p=drop_mlp),
#             nn.Mish(),
#         )

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
        Forward function.

        Args:
            x (torch tensor [batch_size x c x h x w]): Input batch.
            return_fts (bool, Optional): Whether to return encoder features.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
            torch tensor [batch_size x num_features]: Encoder features, if return_fts.
        """
        bs, n_frames, n_landmarks = x["x"].size()

        x_type = self.type_norm(self.type_embed(x["type"]))
        x_landmark = self.landmark_norm(self.landmark_embed(x["landmark"]))
        x_pos = torch.stack([x["x"], x["y"], x["z"]], -1)

#         dists = torch.cat([
#             compute_finger_face_distance(x_pos),
#             compute_hand_features(x_pos, x["type"])
#         ], -1)
#         dists_fts = self.dists_mlp(dists.view(bs * n_frames, -1))
        
        if self.use_cnn:
            x_pos = x_pos.transpose(1, 2).transpose(2, 3).contiguous().view(bs * n_landmarks, -1, n_frames)
            x_pos = self.pos_cnn(x_pos)
            x_pos = x_pos.view(bs, n_landmarks, -1, n_frames).transpose(2, 3).transpose(1, 2).contiguous()
            x_pos = torch.cat([torch.stack([x["x"], x["y"], x["z"]], -1), x_pos], -1)
        else:
            x_pos = add_shift(x_pos)

        x_pos = self.pos_dense(x_pos)

        fts = self.dense(torch.cat([x_type, x_landmark, x_pos], -1))
        
#         fts = fts[:, :self.max_len - 5].contiguous()
#         n_frames = self.max_len - 5

        n_fts = fts.size(-1)
        embed = x["type"][:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)  # this handles padding

        left_hand_fts = fts.view(-1, n_fts)[embed == 1].view(bs * n_frames, -1)
        left_hand_fts = self.left_hand_mlp(left_hand_fts)

        right_hand_fts = fts.view(-1, n_fts)[embed == 2].view(bs * n_frames, -1)
        right_hand_fts = self.right_hand_mlp(right_hand_fts)

        hand_fts = torch.stack([left_hand_fts, right_hand_fts], -1).amax(-1)

        lips_fts = fts.view(-1, n_fts)[embed == 4].view(bs * n_frames, -1)
        lips_fts = self.lips_mlp(lips_fts)

        face_fts = fts.view(-1, n_fts)[
            torch.isin(embed, torch.tensor([3, 6]).to(fts.device))
        ].view(bs * n_frames, -1)

        face_fts = self.face_mlp(face_fts)

#         # Edge fts
#         right_hand_edge_fts = get_edge_features(x, mode="right")
#         right_hand_edge_fts = self.right_hand_edge_dense(right_hand_edge_fts)
#         right_hand_edge_fts = self.right_hand_edge_mlp(right_hand_edge_fts.view(bs * n_frames, -1))

#         left_hand_edge_fts = get_edge_features(x, mode="left")
#         left_hand_edge_fts = self.left_hand_edge_dense(left_hand_edge_fts)
#         left_hand_edge_fts = self.left_hand_edge_mlp(left_hand_edge_fts.view(bs * n_frames, -1))
    
#         hand_edge_fts = torch.stack([left_hand_edge_fts, right_hand_edge_fts], -1).amax(-1)

        fts = fts.view(bs * n_frames, -1)
        fts = self.full_mlp(fts)

        fts = torch.cat([fts, hand_fts, lips_fts, face_fts], -1)  # dists_fts, hand_edge_fts

        fts = self.landmark_mlp(fts)
        fts = fts.view(bs, n_frames, -1)

        mask = x["mask"][:, :, 0][:, :n_frames]
        fts *= mask.unsqueeze(-1)  # probably useless but kept for now

        where = np.random.randint(1, self.transfo_layers + 1) if perm is not None else 0  # Manifold Mixup

        if where == 1:
            fts = coefs.view(-1, 1, 1) * fts + (1 - coefs.view(-1, 1, 1)) * fts[perm]
            mask = ((mask + mask[perm]) > 0).float()

        fts = self.frame_transformer_1(fts, mask).last_hidden_state
        
        if where == 2:
            fts = coefs.view(-1, 1, 1) * fts + (1 - coefs.view(-1, 1, 1)) * fts[perm]
            mask = ((mask + mask[perm]) > 0).float()

        if self.frame_transformer_2 is not None:
            fts = self.frame_transformer_2(fts, mask).last_hidden_state
            
        if where == 3:
            fts = coefs.view(-1, 1, 1) * fts + (1 - coefs.view(-1, 1, 1)) * fts[perm]
            mask = ((mask + mask[perm]) > 0).float()
        
#         print(fts.size())

        if self.frame_transformer_3 is not None:
            fts = self.frame_transformer_3(fts, mask).last_hidden_state

        mask = mask.unsqueeze(-1)
        fts = fts * mask
        fts = fts.sum(1) / mask.sum(1)  # masked avg

        if self.multi_sample_dropout and self.training:
            logits = torch.stack(
                [self.logits(self.dropout(fts)) for _ in range(5)],
                dim=0,
            ).mean(0)
        else:
            logits = self.logits(fts)

        logits_aux = self.logits_aux(fts) if self.num_classes_aux else torch.zeros(1)
        return logits, logits_aux

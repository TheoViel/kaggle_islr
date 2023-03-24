import torch
import torch.nn as nn
import numpy as np

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder

from model_zoo.mha import TransformerBlock
from model_zoo.gcn import GCN
from model_zoo.utils import compute_adjacency_features, add_shift, compute_adjacency_matrix
from utils.torch import load_model_weights


def define_model(
    name,
    embed_dim=256,
    transfo_dim=768,
    transfo_heads=8,
    transfo_layers=4,
    drop_rate=0,
    num_classes=250,
    pretrained_weights="",
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
    if name == "bert_deberta":
        model = SignBertDeberta(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name == "bi_bert":
        model = SignBiBert(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name == "mlp_bert":
        model = SignMLPBert(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            transfo_layers=transfo_layers,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name == "mlp_bert_2":
        model = SignMLPBert2(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            transfo_layers=transfo_layers,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name == "mha_bert":
        model = SignMhaBert(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name =="gcn_bert":
        model = SignGCNBert(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name =="cnn_bert":
        model = SignCNNBert(
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

    return model

    
class SignMLPBert2(nn.Module):
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
        self.landmark_embed = nn.Embedding(101, embed_dim, padding_idx=0)
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
            nn.Linear(embed_dim * 21 , transfo_dim),
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

        self.landmark_mlp = nn.Sequential(
            nn.Linear(embed_dim * n_landmarks + 3 * transfo_dim, transfo_dim * 2),
            nn.BatchNorm1d(transfo_dim * 2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(2 * transfo_dim, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

        name = "microsoft/deberta-v3-base"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim
        config.num_hidden_layers = transfo_layers
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate / 2
        config.hidden_dropout_prob = drop_rate / 2
        config.hidden_act = "relu"
        config.max_relative_positions = 100
        config.position_buckets = 40

        self.frame_transformer = DebertaV2Encoder(config)

        self.logits = nn.Linear(transfo_dim, num_classes)  # * 5

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

#         adj_matrix = self.compute_adjacency_matrix(x_pos)
#         print(adj_matrix.size())

#         adj_fts = compute_adjacency_features(x_pos, x["type"])
#         torch.cat([
#             self.compute_hand_features(x_pos, x['type']),
#             self.compute_hand_to_face_distances(x_pos, x['type'])
#         ], -1)
#         adj_fts = self.adj_dense(adj_fts)
#         adj_fts = add_shift(adj_fts.unsqueeze(-2)).view(bs * n_frames, -1)

        x_pos = add_shift(x_pos)
#         x_pos = self.add_speed(x_pos)
        x_pos = self.pos_dense(x_pos)

#         x_pos = x_pos.transpose(1, 2).contiguous().view(bs * n_landmarks, n_frames, -1).transpose(-1, -2).contiguous()
#         x_pos = self.pos_cnn(x_pos)
#         x_pos = x_pos.transpose(-1, -2).contiguous().view(bs, n_landmarks, n_frames, -1).transpose(1, 2).contiguous()

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

        face_fts = fts.view(-1, n_fts)[torch.isin(embed, torch.tensor([11, 2, 3, 4, 8, 9, 7]).to(fts.device))].view(bs, n_frames, -1, n_fts)
        face_fts = self.face_mlp(face_fts.view(bs * n_frames, -1))

#         print(hand_fts.size())
#         print(lips_fts.size())
#         print(adj_fts.size())
#         print(face_fts.size())
        
        fts = fts.view(-1, n_fts).view(bs, n_frames, -1, n_fts)
#         [
#             embed != 6
# #             ~torch.isin(embed, torch.tensor([6, 11]).to(fts.device))
#         ].view(bs, n_frames, -1, n_fts)  # remove lips
#         print(fts.size())
        fts = fts.view(bs * n_frames, -1)

        fts = torch.cat([fts, hand_fts, lips_fts, face_fts], -1)

        fts = self.landmark_mlp(fts)
        fts = fts.view(bs, n_frames, -1)

        mask = x["mask"][:, :, 0]
#         mask = mask.repeat(self.transfo_heads, 1)
#         mask = (1 - mask) * torch.finfo(mask.dtype).min
#         print(mask.size(), fts.size())
        fts = self.frame_transformer(fts, mask).last_hidden_state

#         fts = self.frame_transformer(
#             fts, mask, output_hidden_states=True
#         ).hidden_states
#         fts = torch.cat(fts, -1)
#         fts = fts.view(bs, n_frames, -1)

        mask = x["mask"][:, :, 0].unsqueeze(-1)
        fts = fts * mask
        fts = fts.sum(1) / mask.sum(1)  # masked avg

        if self.multi_sample_dropout and self.training:
            logits = torch.stack(
                [self.logits(self.dropout(fts)) for _ in range(5)],
                dim=0,
            ).mean(0)
        else:
            logits = self.logits(fts)

        return logits, torch.zeros(1)
    


class SignMLPBert(nn.Module):
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
        self.landmark_embed = nn.Embedding(101, embed_dim, padding_idx=0)
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)

        self.pos_dense = nn.Linear(9, embed_dim, bias=False)
#         self.pos_cnn = nn.Sequential(
#             Conv1dStack(3 * 3, out_dim=embed_dim // 2, kernel_size=5, padding=2),
#             Conv1dStack(embed_dim // 2, out_dim=embed_dim, kernel_size=5, padding=2),
#         )

        self.dense = nn.Linear(3 * embed_dim, embed_dim)
        self.adj_dense = nn.Linear(21 * 21 * 2, transfo_dim // 3)

        self.landmark_mlp = nn.Sequential(
            nn.Linear(embed_dim * n_landmarks + transfo_dim, transfo_dim * 2),
            nn.BatchNorm1d(transfo_dim * 2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(2 * transfo_dim, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

#         self.gcn = GCNAtt(embed_dim, embed_dim)
#         self.conv =  nn.Conv1d(transfo_dim, transfo_dim, kernel_size=3, padding=1, stride=2, bias=False)

        name = "microsoft/deberta-v3-base"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim  # // 2
        config.num_hidden_layers = transfo_layers
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate / 2
        config.hidden_dropout_prob = drop_rate / 2
        config.hidden_act = "relu"
        config.max_relative_positions = 100
        config.position_buckets = 50

        self.frame_transformer = DebertaV2Encoder(config)

        self.logits = nn.Linear(transfo_dim, num_classes)

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

#         adj_matrix = self.compute_adjacency_matrix(x_pos)

        adj_fts = compute_adjacency_features(x_pos, x["type"])
#         torch.cat([
#             self.compute_hand_features(x_pos, x['type']),
#             self.compute_hand_to_face_distances(x_pos, x['type'])
#         ], -1)
        adj_fts = self.adj_dense(adj_fts)
        adj_fts = add_shift(adj_fts.unsqueeze(-2)).view(bs * n_frames, -1)

        x_pos = add_shift(x_pos)
#         x_pos = self.add_speed(x_pos)
        x_pos = self.pos_dense(x_pos)

#         x_pos = x_pos.transpose(1, 2).contiguous().view(bs * n_landmarks, n_frames, -1).transpose(-1, -2).contiguous()
#         x_pos = self.pos_cnn(x_pos)
#         x_pos = x_pos.transpose(-1, -2).contiguous().view(bs, n_landmarks, n_frames, -1).transpose(1, 2).contiguous()

        fts = self.dense(torch.cat([x_type, x_landmark, x_pos], -1))

#         fts = fts.view(bs * n_frames, n_landmarks, -1).transpose(1, 2).contiguous()
#         adj_matrix = adj_matrix.view(bs * n_frames, -1, n_landmarks, n_landmarks)
#         fts = fts + self.gcn(fts, adj_matrix)

        fts = fts.view(bs * n_frames, -1)

        fts = torch.cat([fts, adj_fts], -1)

        fts = self.landmark_mlp(fts)
        fts = fts.view(bs, n_frames, -1)

#         fts = fts.transpose(1, 2)
#         fts = self.conv(fts)
#         fts = fts.transpose(1, 2)

        mask = x["mask"][:, :, 0]
        fts = self.frame_transformer(fts, mask).last_hidden_state

#         fts = self.frame_transformer(fts, mask, output_hidden_states=True).hidden_states
#         fts = torch.cat(fts, -1)

        mask = x["mask"][:, :, 0].unsqueeze(-1)
        fts = fts * mask
        fts = fts.sum(1) / mask.sum(1)  # masked avg

        if self.multi_sample_dropout and self.training:
            logits = torch.stack(
                [self.logits(self.dropout(fts)) for _ in range(5)],
                dim=0,
            ).mean(0)
        else:
            logits = self.logits(fts)

        return logits, torch.zeros(1)


class SignGCNBert(nn.Module):
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

        self.type_embed = nn.Embedding(12, embed_dim, padding_idx=0)
        self.landmark_embed = nn.Embedding(101, embed_dim, padding_idx=0)
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)

        self.frames = nn.parameter.Parameter(
            torch.tensor([i for i in range(100)]), requires_grad=False
        )
        self.frame_embed = nn.Embedding(200, transfo_dim, padding_idx=0)
        self.frame_norm = nn.LayerNorm(transfo_dim)

        #         self.cls_embed = nn.Parameter(torch.zeros((1, transfo_dim)))
        self.pos_dense = nn.Linear(9, embed_dim, bias=False)
        self.dense = nn.Linear(3 * embed_dim, embed_dim)

        self.landmark_gcn = GCN(embed_dim, embed_dim, dropout=0)

        self.landmark_mlp = nn.Sequential(
            nn.Linear(embed_dim * n_landmarks, transfo_dim * 2),
            nn.BatchNorm1d(transfo_dim * 2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(2 * transfo_dim, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

        name = "bert-base-uncased"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim * 2
        config.num_hidden_layers = 4
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate
        config.hidden_dropout_prob = drop_rate
        config.hidden_act = "relu"

        self.frame_transformer = BertEncoder(config)

        self.logits = nn.Linear(transfo_dim, num_classes)  # * 5

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
        x_type = self.type_norm(self.type_embed(x["type"]))
        x_landmark = self.landmark_norm(self.landmark_embed(x["landmark"]))

        x_pos = torch.stack([x["x"], x["y"], x["z"]], -1)
        adj = compute_adjacency_matrix(x_pos)

        x_pos = add_shift(x_pos)
        #         x_pos = self.add_speed(x_pos)
        x_pos = self.pos_dense(x_pos)

        fts = self.dense(torch.cat([x_type, x_landmark, x_pos], -1))

        bs, n_frames, n_landmarks, _ = fts.size()

        fts = fts.view(bs * n_frames, n_landmarks, -1)
        adj = adj.view(bs * n_frames, n_landmarks, n_landmarks)

        fts = self.landmark_gcn(fts, adj)

        fts = fts.view(bs * n_frames, -1)
        fts = self.landmark_mlp(fts)

        fts = fts.view(bs, n_frames, -1)

        #         fts = torch.cat([fts, self.cls_embed.unsqueeze(0).repeat(fts.size(0), 1, 1)], 1)
        frames = self.frames[: fts.size(1)].unsqueeze(0)
        x_frame = self.frame_norm(self.frame_embed(frames)).repeat(fts.size(0), 1, 1)
        fts = fts + x_frame

        # Mask
        mask = 1 - x["mask"][:, :, 0]
        att_mask = mask.unsqueeze(1).unsqueeze(1) * torch.finfo(mask.dtype).min
        encoder_att_mask = mask * torch.finfo(mask.dtype).min

        fts = self.frame_transformer(
            fts, attention_mask=att_mask, encoder_attention_mask=encoder_att_mask
        ).last_hidden_state
        fts = fts.view(bs, n_frames, -1)

        mask = x["mask"][:, :, 0].unsqueeze(-1)
        fts = fts * mask
        fts = fts.sum(1) / mask.sum(1)  # masked avg

        logits = self.logits(fts)

        return logits, torch.zeros(1)


class SignMhaBert(nn.Module):
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

        self.type_embed = nn.Embedding(12, embed_dim, padding_idx=0)
        self.landmark_embed = nn.Embedding(101, embed_dim, padding_idx=0)
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)

        self.cls_embed = nn.Parameter(torch.zeros((1, embed_dim)))

        self.pos_dense = nn.Linear(3, embed_dim, bias=False)
        self.dense = nn.Linear(3 * embed_dim, embed_dim)

        self.landmark_transfo = TransformerBlock(
            embed_dim, embed_dim, num_head=transfo_heads
        )

        self.transition_dense = nn.Linear(embed_dim, transfo_dim)

        self.frames = nn.parameter.Parameter(
            torch.tensor([i for i in range(100)]), requires_grad=False
        )
        self.frame_embed = nn.Embedding(200, transfo_dim, padding_idx=0)
        self.frame_norm = nn.LayerNorm(transfo_dim)

        name = "bert-base-uncased"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim * 2
        config.num_hidden_layers = 4
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate
        config.hidden_dropout_prob = drop_rate
        config.hidden_act = "relu"

        self.frame_transformer = BertEncoder(config)

        self.logits = nn.Linear(transfo_dim, num_classes)

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
        x_type = self.type_norm(self.type_embed(x["type"]))
        x_landmark = self.landmark_norm(self.landmark_embed(x["landmark"]))
        x_pos = self.pos_dense(torch.stack([x["x"], x["y"], x["z"]], -1))

        #         fts = x_type + x_landmark + x_pos
        fts = self.dense(torch.cat([x_type, x_landmark, x_pos], -1))

        bs, n_frames, n_landmarks, _ = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

        fts = torch.cat([self.cls_embed.unsqueeze(0).repeat(fts.size(0), 1, 1), fts], 1)

        #         print(fts.size())
        fts = self.landmark_transfo(fts)
        #         print(fts.size())

        fts = fts.view(bs, n_frames, -1)
        fts = self.transition_dense(fts)

        #         fts = torch.cat([fts, self.cls_embed.unsqueeze(0).repeat(fts.size(0), 1, 1)], 1)
        frames = self.frames[: fts.size(1)].unsqueeze(0)
        x_frame = self.frame_norm(self.frame_embed(frames)).repeat(fts.size(0), 1, 1)
        fts = fts + x_frame

        # Mask
        mask = 1 - x["mask"][:, :, 0]
        att_mask = mask.unsqueeze(1).unsqueeze(1) * torch.finfo(mask.dtype).min
        encoder_att_mask = mask * torch.finfo(mask.dtype).min

        fts = self.frame_transformer(
            fts, attention_mask=att_mask, encoder_attention_mask=encoder_att_mask
        ).last_hidden_state
        fts = fts.view(bs, n_frames, -1)

        mask = x["mask"][:, :, 0].unsqueeze(-1)
        fts = fts * mask
        fts = fts.sum(1) / mask.sum(1)  # masked avg

        logits = self.logits(fts)

        return logits, torch.zeros(1)


class SignBertDeberta(nn.Module):
    """
    Model with an attention mechanism.
    """

    def __init__(
        self,
        embed_dim=256,
        transfo_dim=768,
        transfo_heads=1,
        num_classes=250,
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

        self.type_embed = nn.Embedding(12, embed_dim, padding_idx=0)
        self.landmark_embed = nn.Embedding(101, embed_dim, padding_idx=0)

        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)

        self.pos_dense = nn.Linear(3, embed_dim)
        #         self.dense = nn.Linear(3 * embed_dim, transfo_dim)

        name = "bert-base-uncased"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim * 2
        config.num_hidden_layers = 4
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate
        config.hidden_dropout_prob = drop_rate

        self.landmark_transformer = BertEncoder(config)

        name = "microsoft/deberta-v3-base"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim * 2
        config.num_hidden_layers = 4
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate
        config.hidden_dropout_prob = drop_rate
        config.max_relative_positions = 100

        self.frame_transformer = DebertaV2Encoder(config)

        self.logits = nn.Linear(transfo_dim, num_classes)

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
        x_type = self.type_norm(self.type_embed(x["type"]))
        x_landmark = self.landmark_norm(self.landmark_embed(x["landmark"]))
        x_pos = self.pos_dense(torch.stack([x["x"], x["y"], x["z"]], -1))

        fts = torch.cat([x_type, x_landmark, x_pos], -1)
        #         fts = x_type + x_landmark + x_pos

        bs, n_frames, n_landmarks, _ = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

        fts = self.landmark_transformer(fts).last_hidden_state
        fts = fts.view(bs, n_frames, n_landmarks, -1)
        fts = fts.mean(2)

        #         hidden_states = self.frame_transformer(fts, x['mask'][:, :, 0]).hidden_states
        #         fts = torch.cat(hidden_states[::-1], -1)
        fts = self.frame_transformer(fts, x["mask"][:, :, 0]).last_hidden_state
        fts = fts.view(bs, n_frames, -1)
        fts = fts.mean(1)

        logits = self.logits(fts)

        return logits, torch.zeros(1)

    def get_landmark_attentions(self, x):
        x_type = self.type_norm(self.type_embed(x["type"]))
        x_landmark = self.landmark_norm(self.landmark_embed(x["landmark"]))
        x_pos = self.pos_dense(torch.stack([x["x"], x["y"], x["z"]], -1))

        fts = torch.cat([x_type, x_landmark, x_pos], -1)

        bs, n_frames, n_landmarks, _ = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

        out = self.landmark_transformer(fts, output_attentions=True)

        return out.attentions, out.cross_attentions

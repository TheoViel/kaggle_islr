import torch
import torch.nn as nn
import numpy as np

from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder

from model_zoo.mha import TransformerBlock
from model_zoo.gcn import GCN
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


def positional_encoding(length, embed_dim):
    dim = embed_dim//2

    position = np.arange(length)[:, np.newaxis]     # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :]/dim   # (1, dim)

    angle = 1 / (10000**dim)         # (1, dim)
    angle = position * angle    # (pos, dim)

    pos_embed = np.concatenate(
        [np.sin(angle), np.cos(angle)],
        axis=-1
    )
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed


class Conv1dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv1dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h


class Conv2dStack(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, padding=1, dilation=1):
        super(Conv2dStack, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(),
        )
        self.res = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        h = self.res(x)
        return x + h

    
class GCNAtt(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super(GCNAtt, self).__init__()
        self.conv = Conv1dStack(in_channel, out_channel, 3, padding=1)
        self.mat_conv = Conv2dStack(5, out_channel, kernel_size=1, padding=0)

    def forward(self, x, matrix):
        x = self.conv(x)
        matrix = self.mat_conv(matrix)
        x = torch.matmul(matrix, x.unsqueeze(-1))
        return x.squeeze(-1)

    
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

#         self.frames = nn.parameter.Parameter(torch.tensor([i for i in range(100)]), requires_grad=False)
#         self.frame_embed = nn.Embedding(100, transfo_dim, padding_idx=0)
#         self.frame_norm = nn.LayerNorm(transfo_dim)

#         self.cls_embed = nn.Parameter(torch.zeros((1, transfo_dim)))
        self.pos_dense = nn.Linear(9, embed_dim, bias=False)
#         self.pos_cnn = nn.Sequential(
#             Conv1dStack(3 * 3, out_dim=embed_dim // 2, kernel_size=5, padding=2),
#             Conv1dStack(embed_dim // 2, out_dim=embed_dim, kernel_size=5, padding=2),
#         )

        self.dense = nn.Linear(3 * embed_dim, embed_dim)
        
#         self.gcn = GCNAtt(embed_dim, embed_dim)
        
        self.adj_dense = nn.Sequential(
            nn.Linear(21 * 21 * 2, transfo_dim // 3),
        )

        self.landmark_mlp = nn.Sequential(
            nn.Linear(embed_dim * n_landmarks + transfo_dim, transfo_dim * 2),  # transfo_dim
            nn.BatchNorm1d(transfo_dim * 2),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
            nn.Linear(2 * transfo_dim, transfo_dim),
            nn.BatchNorm1d(transfo_dim),
            nn.Dropout(p=drop_rate),
            nn.LeakyReLU(),
        )

#         name = "bert-base-uncased"
#         config = AutoConfig.from_pretrained(name, output_hidden_states=True)
#         config.hidden_size = transfo_dim
#         config.intermediate_size = transfo_dim * 2
#         config.num_hidden_layers = 4
#         config.num_attention_heads = transfo_heads
#         config.attention_probs_dropout_prob = drop_rate
#         config.hidden_dropout_prob = drop_rate
#         config.hidden_act = "relu"
#         self.frame_transformer = BertEncoder(config)

        name = "microsoft/deberta-v3-base"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim # // 2
        config.num_hidden_layers = transfo_layers
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate / 2
        config.hidden_dropout_prob = drop_rate / 2
        config.hidden_act = "relu"
        config.max_relative_positions = 100
        config.position_buckets = 50

        self.frame_transformer = DebertaV2Encoder(config)

#         self.dropout = nn.Dropout(0.5)
        self.logits = nn.Linear(transfo_dim, num_classes)   # * 5
        
    @staticmethod
    def compute_adjacency_features(x, embed):
        bs, n_frames, n_landmarks, n_fts = x.size()
        embed = embed[:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)  # this avoids padding
        left_hand = x.view(-1, n_fts)[embed == 5].view(bs, n_frames, -1, n_fts)
        adj_left_hand = ((left_hand.unsqueeze(-2) - left_hand.unsqueeze(-3)) ** 2).sum(-1).sqrt()

        right_hand = x.view(-1, n_fts)[embed == 6].view(bs, n_frames, -1, n_fts)
        adj_right_hand = ((right_hand.unsqueeze(-2) - right_hand.unsqueeze(-3)) ** 2).sum(-1).sqrt()

        return torch.cat([adj_left_hand.view(bs, n_frames, -1), adj_right_hand.view(bs, n_frames, -1)], -1)
    
    @staticmethod
    def compute_adjacency_matrix(x_pos):
        mask = ((x_pos == 0).sum(-1) != 3)
        adj = ((x_pos.unsqueeze(-2) - x_pos.unsqueeze(-3)) ** 2).sum(-1).sqrt()  # distance matrix
        adj = 1 / (adj + 1)
        adj = adj * mask.unsqueeze(-1) * mask.unsqueeze(-2)

        adj = torch.stack([adj, torch.ones_like(adj), (adj > 0.5).float(), (adj > 0.75).float(), (adj > 0.9).float()], 2)
        return adj

    @staticmethod
    def compute_hand_features(x, embed):
        bs, n_frames, n_landmarks, n_fts = x.size()
        embed = embed[:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)  # this avoids padding
        left_hand = x.view(-1, n_fts)[embed == 5].view(bs, n_frames, -1, n_fts)
        adj_left_hand = ((left_hand.unsqueeze(-2) - left_hand.unsqueeze(-3)) ** 2).sum(-1).sqrt()

        sz = adj_left_hand.size(3)
        adj_left_hand = adj_left_hand.view(bs * n_frames, sz, -1)
        ids_a, ids_b = torch.triu_indices(sz, sz, offset=1).unbind()
        adj_left_hand = adj_left_hand[:, ids_a, ids_b].view(bs, n_frames, -1)

        right_hand = x.view(-1, n_fts)[embed == 6].view(bs, n_frames, -1, n_fts)
        adj_right_hand = ((right_hand.unsqueeze(-2) - right_hand.unsqueeze(-3)) ** 2).sum(-1).sqrt()
        
        sz = adj_right_hand.size(3)
        adj_right_hand = adj_right_hand.view(bs * n_frames, sz, -1)
        ids_a, ids_b = torch.triu_indices(sz, sz, offset=1).unbind()
        adj_right_hand = adj_right_hand[:, ids_a, ids_b].view(bs, n_frames, -1)

        return torch.cat([adj_left_hand, adj_right_hand], -1)
    
    @staticmethod
    def compute_hand_to_face_distances(x, embed):
        bs, n_frames, n_landmarks, n_fts = x.size()
        embed = embed[:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)  # this avoids padding
    
        face = x.view(-1, n_fts)[torch.isin(embed, torch.tensor([3, 4, 8, 9, 7]).to(x.device))].view(bs, n_frames, -1, n_fts)

        left_hand = x.view(-1, n_fts)[embed == 5].view(bs, n_frames, -1, n_fts)
        right_hand = x.view(-1, n_fts)[embed == 6].view(bs, n_frames, -1, n_fts)
        
        left_dists = ((left_hand.unsqueeze(-2) - face.unsqueeze(-3)) ** 2).sum(-1).sqrt()
        right_dists = ((right_hand.unsqueeze(-2) - face.unsqueeze(-3)) ** 2).sum(-1).sqrt()
        
        return torch.cat([left_dists, right_dists], -1).view(bs, n_frames, -1)

    @staticmethod
    def add_shift(x, n=1):
        padding = torch.zeros((x.size(0), n, x.size(2), x.size(3)), device=x.device)
        x = torch.cat([
            torch.cat([x[:, n:], padding], axis=1),
            x,
            torch.cat([padding, x[:, :-n]], axis=1),
        ], axis=3)
        return x
    
    @staticmethod
    def add_speed(x, n=1):
        padding = torch.zeros((x.size(0), n, x.size(2), x.size(3)), device=x.device)
        x = torch.cat([
            torch.cat([x[:, n:], padding], axis=1) - x,
            x,
            x - torch.cat([padding, x[:, :-n]], axis=1),
        ], axis=3)
        return x

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
        bs, n_frames, n_landmarks = x['x'].size()

        x_type = self.type_norm(self.type_embed(x['type']))
        x_landmark = self.landmark_norm(self.landmark_embed(x['landmark']))
        x_pos = torch.stack([x['x'], x['y'], x['z']], -1)
        
#         adj_matrix = self.compute_adjacency_matrix(x_pos)
#         print(adj_matrix.size())

        adj_fts = self.compute_adjacency_features(x_pos, x['type'])
#         torch.cat([
#             self.compute_hand_features(x_pos, x['type']), 
#             self.compute_hand_to_face_distances(x_pos, x['type'])
#         ], -1)
        adj_fts = self.adj_dense(adj_fts)
        adj_fts = self.add_shift(adj_fts.unsqueeze(-2)).view(bs * n_frames, -1)

        x_pos = self.add_shift(x_pos)
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
        
        mask = x['mask'][:, :, 0]
#         mask = mask.repeat(self.transfo_heads, 1)
#         mask = (1 - mask) * torch.finfo(mask.dtype).min
#         print(mask.size(), fts.size())
        fts = self.frame_transformer(fts, mask).last_hidden_state
    
#         print(fts.size())
#         print(fts.max(-1)[0])
        
# #         fts = torch.cat([fts, self.cls_embed.unsqueeze(0).repeat(fts.size(0), 1, 1)], 1)
#         frames = self.frames[:fts.size(1)].unsqueeze(0)
#         x_frame = self.frame_norm(self.frame_embed(frames)).repeat(fts.size(0), 1, 1)
#         fts = fts + x_frame

#         # Mask
#         mask = 1 - x['mask'][:, :, 0]
#         att_mask = mask.unsqueeze(1).unsqueeze(1) * torch.finfo(mask.dtype).min
#         encoder_att_mask = mask * torch.finfo(mask.dtype).min

# #         fts = self.frame_transformer(
# #             fts, attention_mask=att_mask, encoder_attention_mask=encoder_att_mask, output_hidden_states=True
# #         ).hidden_states
# #         fts = torch.cat(fts, -1)
#         fts = self.frame_transformer(fts, attention_mask=att_mask, encoder_attention_mask=encoder_att_mask).last_hidden_state
#         fts = fts.view(bs, n_frames, -1)

        mask = x['mask'][:, :, 0].unsqueeze(-1)
        fts = fts * mask
        fts = fts.sum(1) / mask.sum(1)  # masked avg
        
        if self.multi_sample_dropout and self.training:
            logits = torch.stack(
                [self.logits(self.dropout(fts)) for _ in range(5)], dim=0,
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

        self.frames = nn.parameter.Parameter(torch.tensor([i for i in range(100)]), requires_grad=False)
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

        self.logits = nn.Linear(transfo_dim, num_classes)   # * 5
        
    @staticmethod
    def add_shift(x, n=1):
        padding = torch.zeros((x.size(0), n, x.size(2), x.size(3)), device=x.device)
        x = torch.cat([
            torch.cat([x[:, n:], padding], axis=1),
            x,
            torch.cat([padding, x[:, :-n]], axis=1),
        ], axis=3)
        return x

    @staticmethod
    def compute_adjacency_matrix(x_pos):
        mask = ((x_pos == 0).sum(-1) != 3)
        adj = ((x_pos.unsqueeze(-2) - x_pos.unsqueeze(-3)) ** 2).sum(-1).sqrt()  # distance matrix
        adj = 1 / (adj + 1)
        adj = adj * mask.unsqueeze(-1) * mask.unsqueeze(-2)
        return adj


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
        x_type = self.type_norm(self.type_embed(x['type']))
        x_landmark = self.landmark_norm(self.landmark_embed(x['landmark']))

        x_pos = torch.stack([x['x'], x['y'], x['z']], -1)
        adj = self.compute_adjacency_matrix(x_pos)

        x_pos = self.add_shift(x_pos)
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
        frames = self.frames[:fts.size(1)].unsqueeze(0)
        x_frame = self.frame_norm(self.frame_embed(frames)).repeat(fts.size(0), 1, 1)
        fts = fts + x_frame

        # Mask
        mask = 1 - x['mask'][:, :, 0]
        att_mask = mask.unsqueeze(1).unsqueeze(1) * torch.finfo(mask.dtype).min
        encoder_att_mask = mask * torch.finfo(mask.dtype).min

        fts = self.frame_transformer(fts, attention_mask=att_mask, encoder_attention_mask=encoder_att_mask).last_hidden_state
        fts = fts.view(bs, n_frames, -1)

        mask = x['mask'][:, :, 0].unsqueeze(-1)
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

        self.landmark_transfo = TransformerBlock(embed_dim, embed_dim, num_head=transfo_heads)

        self.transition_dense = nn.Linear(embed_dim, transfo_dim)

        self.frames = nn.parameter.Parameter(torch.tensor([i for i in range(100)]), requires_grad=False)
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

        self.logits = nn.Linear(transfo_dim , num_classes)

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
        x_type = self.type_norm(self.type_embed(x['type']))
        x_landmark = self.landmark_norm(self.landmark_embed(x['landmark']))
        x_pos = self.pos_dense(torch.stack([x['x'], x['y'], x['z']], -1))

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
        frames = self.frames[:fts.size(1)].unsqueeze(0)
        x_frame = self.frame_norm(self.frame_embed(frames)).repeat(fts.size(0), 1, 1)
        fts = fts + x_frame

        # Mask
        mask = 1 - x['mask'][:, :, 0]
        att_mask = mask.unsqueeze(1).unsqueeze(1) * torch.finfo(mask.dtype).min
        encoder_att_mask = mask * torch.finfo(mask.dtype).min

        fts = self.frame_transformer(fts, attention_mask=att_mask, encoder_attention_mask=encoder_att_mask).last_hidden_state
        fts = fts.view(bs, n_frames, -1)

        mask = x['mask'][:, :, 0].unsqueeze(-1)
        fts = fts * mask
        fts = fts.sum(1) / mask.sum(1)  # masked avg
        
        logits = self.logits(fts)

        return logits, torch.zeros(1)


class SignBiBert(nn.Module):
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
        self.frame_embed = nn.Embedding(200, transfo_dim, padding_idx=0)
        
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)
        self.frame_norm = nn.LayerNorm(transfo_dim)
    
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

        self.frame_transformer = BertEncoder(config)

        self.logits = nn.Linear(transfo_dim , num_classes)

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
        x_type = self.type_norm(self.type_embed(x['type']))
        x_landmark = self.landmark_norm(self.landmark_embed(x['landmark']))
        x_pos = self.pos_dense(torch.stack([x['x'], x['y'], x['z']], -1))

        fts = torch.cat([x_type, x_landmark, x_pos], -1)

        bs, n_frames, n_landmarks, _ = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

        fts = self.landmark_transformer(fts).last_hidden_state
        fts = fts.view(bs, n_frames, n_landmarks, -1)
        fts = fts.mean(2)

        frames = (torch.arange(fts.size(1)) + 1).unsqueeze(0).to(x['x'].device)
        x_frame = self.frame_norm(self.frame_embed(frames)).repeat(fts.size(0), 1, 1)
        fts = fts + x_frame
        
        # Mask
        mask = 1 - x['mask'][:, :, 0]
        att_mask = mask.unsqueeze(1).unsqueeze(1) * torch.finfo(mask.dtype).min
        encoder_att_mask = mask * torch.finfo(mask.dtype).min

        fts = self.frame_transformer(fts, attention_mask=att_mask, encoder_attention_mask=encoder_att_mask).last_hidden_state
        fts = fts.view(bs, n_frames, -1)

        mask = x['mask'][:, :, 0].unsqueeze(-1)
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

        self.logits = nn.Linear(transfo_dim , num_classes)

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
        x_type = self.type_norm(self.type_embed(x['type']))
        x_landmark = self.landmark_norm(self.landmark_embed(x['landmark']))
        x_pos = self.pos_dense(torch.stack([x['x'], x['y'], x['z']], -1))

        fts = torch.cat([x_type, x_landmark, x_pos], -1)
#         fts = x_type + x_landmark + x_pos

        bs, n_frames, n_landmarks, _ = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

        fts = self.landmark_transformer(fts).last_hidden_state
        fts = fts.view(bs, n_frames, n_landmarks, -1)
        fts = fts.mean(2)

#         hidden_states = self.frame_transformer(fts, x['mask'][:, :, 0]).hidden_states
#         fts = torch.cat(hidden_states[::-1], -1)
        fts = self.frame_transformer(fts, x['mask'][:, :, 0]).last_hidden_state
        fts = fts.view(bs, n_frames, -1)
        fts = fts.mean(1)
        
        logits = self.logits(fts)

        return logits, torch.zeros(1)

    def get_landmark_attentions(self, x):
        x_type = self.type_norm(self.type_embed(x['type']))
        x_landmark = self.landmark_norm(self.landmark_embed(x['landmark']))
        x_pos = self.pos_dense(torch.stack([x['x'], x['y'], x['z']], -1))
       
        fts = torch.cat([x_type, x_landmark, x_pos], -1)

        bs, n_frames, n_landmarks, _ = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

        out = self.landmark_transformer(fts, output_attentions=True)
        
        return out.attentions, out.cross_attentions

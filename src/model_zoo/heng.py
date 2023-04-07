# class Config(object):
if 1:
    num_class = 250
    max_length = 256
    point_dim = 1302
    embed_dim = 384
    num_head = 4
    num_block = 1
    label_smoothing = 0.75
# CFG = Config()


import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F


def pack_seq(
    seq,
):
    length = [min(len(s), max_length) for s in seq]
    batch_size = len(seq)
    K = seq[0].shape[1]
    L = max(length)
    # print(length)

    x = torch.zeros((batch_size, L, point_dim)).to(seq[0].device)
    x_mask = torch.zeros((batch_size, L)).to(seq[0].device)
    for b in range(batch_size):
        l = length[b]
        x[b, :l] = seq[b][:l, :]
        x_mask[b, l:] = 1
    x_mask = x_mask > 0.5

    return x, x_mask


def positional_encoding(length, embed_dim):
    dim = embed_dim // 2
    position = np.arange(length)[:, np.newaxis]  # (seq, 1)
    dim = np.arange(dim)[np.newaxis, :] / dim  # (1, dim)
    angle = 1 / (10000**dim)  # (1, dim)
    angle = position * angle  # (pos, dim)
    pos_embed = np.concatenate([np.sin(angle), np.cos(angle)], axis=-1)
    pos_embed = torch.from_numpy(pos_embed).float()
    return pos_embed


class XEmbed(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(point_dim, embed_dim * 2, bias=True),
            nn.LayerNorm(embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim, bias=True),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, x_mask):
        B, L, _ = x.shape
        v = self.v(x)
        x = v
        return x, x_mask


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_head,
        out_dim,
    ):
        super().__init__()
        self.attn = MyMultiHeadAttention(
            embed_dim=embed_dim,
            out_dim=embed_dim,
            qk_dim=embed_dim // num_head,
            v_dim=embed_dim // num_head,
            num_head=num_head,
        )
        self.ffn = FeedForward(embed_dim, out_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x, x_mask=None):
        x = x + self.attn((self.norm1(x)), x_mask)
        x = x + self.ffn((self.norm2(x)))
        return x


class MyMultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        out_dim,
        qk_dim,
        v_dim,
        num_head,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.qk_dim = qk_dim
        self.v_dim = v_dim

        self.q = nn.Linear(embed_dim, qk_dim * num_head)
        self.k = nn.Linear(embed_dim, qk_dim * num_head)
        self.v = nn.Linear(embed_dim, v_dim * num_head)

        self.out = nn.Linear(v_dim * num_head, out_dim)
        self.scale = 1 / (qk_dim**0.5)

    # https://github.com/pytorch/pytorch/issues/40497
    def forward(self, x, x_mask):
        B, L, dim = x.shape
        # out, _ = self.mha(x,x,x, key_padding_mask=x_mask)
        num_head = self.num_head
        qk_dim = self.qk_dim
        v_dim = self.v_dim

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.reshape(B, L, num_head, qk_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, L, num_head, qk_dim).permute(0, 2, 3, 1).contiguous()
        v = v.reshape(B, L, num_head, v_dim).permute(0, 2, 1, 3).contiguous()

        dot = torch.matmul(q, k) * self.scale  # H L L
        x_mask = x_mask.reshape(B, 1, 1, L).expand(-1, num_head, L, -1)
        # dot[x_mask]= -1e4
        dot.masked_fill_(x_mask, -1e4)
        attn = F.softmax(dot, -1)  # L L

        v = torch.matmul(attn, v)  # L H dim
        v = v.permute(0, 2, 1, 3).reshape(B, L, v_dim * num_head).contiguous()
        out = self.out(v)

        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.mlp(x)

    
    
class HCKTransfo(nn.Module):
    def __init__(self, dim=512, num_head=4, p=0.4, max_length=40, num_block=1):
        super().__init__()

        pos_embed = positional_encoding(max_length, dim)
        self.pos_embed = nn.Parameter(pos_embed)
        self.cls_embed = nn.Parameter(torch.zeros((1, dim)))

        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    num_head,
                    dim,
                )
                for i in range(num_block)
            ]
        )
        self.dropout = nn.Dropout(p)

    def forward(self, x, mask):
        """
        Mask expects False for used tokens, True for others.
        """
        B, L, _ = x.shape

        # Add position embedding
        x = x + self.pos_embed[:L].unsqueeze(0)
        
        # Concat CLS token
        x = torch.cat([self.cls_embed.unsqueeze(0).repeat(B, 1, 1), x], 1)
        mask = torch.cat([torch.zeros(B, 1).to(mask), mask], 1)

        # Transfo
        for block in self.encoder:
            x = block(x, mask)
            
        # Dropout
        x = self.dropout(x)
        
        return x

    

class Net(nn.Module):
    def __init__(self, num_class=num_class):
        super().__init__()
        self.output_type = ["inference", "loss"]

        self.x_embed = XEmbed()

        pos_embed = positional_encoding(max_length, embed_dim)
        self.pos_embed = nn.Parameter(pos_embed)
        self.cls_embed = nn.Parameter(torch.zeros((1, embed_dim)))

        self.encoder = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim,
                    num_head,
                    embed_dim,
                )
                for i in range(num_block)
            ]
        )
        self.logit = nn.Linear(embed_dim, num_class)

    def forward(self, batch):
        xyz = batch["xyz"]

        # ----
        x, x_mask = pack_seq(xyz)
        x, x_mask = self.x_embed(x, x_mask)
        B, L, _ = x.shape

        x = x + self.pos_embed[:L].unsqueeze(0)
        x = torch.cat([self.cls_embed.unsqueeze(0).repeat(B, 1, 1), x], 1)
        print(x_mask)
        x_mask = torch.cat([torch.zeros(B, 1).to(x_mask), x_mask], 1)

        for block in self.encoder:
            x = block(x, x_mask)
        x = F.dropout(x, p=0.4, training=self.training)

        # ---
        # mask pool
        x_mask = x_mask.unsqueeze(-1)
        x_mask = 1 - x_mask.float()
        last = (x * x_mask).sum(1) / x_mask.sum(1)
        logit = self.logit(last)

        output = {}
        if "loss" in self.output_type:
            output["label_loss"] = F.cross_entropy(
                logit, batch["label"], label_smoothing=label_smoothing
            )  # 0.5

        if "inference" in self.output_type:
            output["sign"] = torch.softmax(logit, -1)

        return output

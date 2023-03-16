import torch
import torch.nn as nn
import torch.nn.functional as F


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


# https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_head,
        batch_first,
    ):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )

    def forward(self, x, x_mask):
        out, _ = self.mha(x, x, x, key_padding_mask=x_mask)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head, batch_first=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads=num_head,
            bias=True,
            add_bias_kv=False,
            kdim=None,
            vdim=None,
            dropout=0.0,
            batch_first=batch_first,
        )
        self.embed_dim = embed_dim
        self.num_head = num_head
#         self.

    # https://github.com/pytorch/text/blob/60907bf3394a97eb45056a237ca0d647a6e03216/torchtext/modules/multiheadattention.py#L5
    def forward(self, x):
#         print(self.mha.in_proj_weight.size())
        bs = x.size(0)
        q = F.linear(
            x[:, :1], self.mha.in_proj_weight[:self.embed_dim], self.mha.in_proj_bias[:self.embed_dim]
        )  # since we need only cls
        k = F.linear(
            x, self.mha.in_proj_weight[self.embed_dim: self.embed_dim * 2], self.mha.in_proj_bias[self.embed_dim: self.embed_dim * 2]
        )
        v = F.linear(
            x, self.mha.in_proj_weight[self.embed_dim * 2:], self.mha.in_proj_bias[self.embed_dim * 2:]
        )

#         print(q.size(), k.size(), v.size())
        
        q = q.reshape(bs, -1, self.num_head, self.embed_dim // self.num_head).permute(0, 2, 1, 3)  # BS x n_heads x 1 x n_fts
        k = k.reshape(bs, -1, self.num_head, self.embed_dim // self.num_head).permute(0, 2, 3, 1)  # BS x n_heads x n_fts x landmarks
        v = v.reshape(bs, -1, self.num_head, self.embed_dim // self.num_head).permute(0, 2, 1, 3)  # BS x n_heads x landmarks x n_fts
        
#         print(q.size(), k.size(), v.size())
        
        dot = torch.matmul(q, k) * (1 / (self.embed_dim // self.num_head)**0.5)
        attn = F.softmax(dot, -1)  # BS x n_heads x 1 x landmarks
        
#         print(attn.size())
        
        out = torch.matmul(attn, v)  # BS x n_heads x landmarks x n_fts
        
#         print(out.size())

        out = out.reshape(bs, self.embed_dim)  # BS x n_fts*n_heads
        out = F.linear(out, self.mha.out_proj.weight, self.mha.out_proj.bias)  # BS x n_fts*n_heads
        
#         print("out", out.size())
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, out_dim, num_head=8, batch_first=True):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_head,batch_first)
        self.ffn   = FeedForward(embed_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x[:, 0] + self.attn((self.norm1(x)))
        x = x + self.ffn((self.norm2(x)))
        return x


# coding=utf-8
# Copyright 2020 Microsoft and the Hugging Face Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optimized Deberta"""

import nobuco
import numpy as np
import torch
import torch.utils.checkpoint

from torch import nn
from torch.nn import LayerNorm
from collections.abc import Sequence
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import ConvLayer


# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaLayerNorm->LayerNorm
class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaAttention with Deberta->DebertaV2
class DebertaV2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        ids=None,
        ids_t=None
    ):
        self_output = self.self(
            hidden_states,
            #             attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            ids=ids,
            ids_t=ids_t,
        )
        if output_attentions:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->DebertaV2
class DebertaV2Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaOutput with DebertaLayerNorm->LayerNorm
class DebertaV2Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# Copied from transformers.models.deberta.modeling_deberta.DebertaLayer with Deberta->DebertaV2
class DebertaV2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
        ids=None,
        ids_t=None
    ):
        attention_output = self.attention(
            hidden_states,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
            ids=ids,
            ids_t=ids_t
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class DebertaV2Encoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()

        self.layer = nn.ModuleList(
            [DebertaV2Layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        self.norm_rel_ebd = [
            x.strip()
            for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")
        ]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(
                config.hidden_size, config.layer_norm_eps, elementwise_affine=True
            )

        self.conv = (
            ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        )
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
        ids=None,
        ids_t=None
    ):

        relative_pos = None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        output_states = next_kv
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            output_states = layer_module(
                next_kv,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
                ids=ids,
                ids_t=ids_t,
            )

            if output_attentions:
                output_states, att_m = output_states


            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (output_states,)

        if not return_dict:
            return tuple(
                v
                for v in [output_states, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module
    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]
    """

    def __init__(self, config, ids=None, ids_t=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(
            config, "attention_head_size", _attention_head_size
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = (
            config.pos_att_type if config.pos_att_type is not None else []
        )
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(
                        config.hidden_size, self.all_head_size, bias=True
                    )
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(
                        config.hidden_size, self.all_head_size
                    )

        self.ids = torch.zeros(
            (
                config.max_len + 1,
                config.num_attention_heads,
                config.max_len,
                config.max_len,
            ),
            dtype=torch.int,
        )
        self.ids_t = torch.zeros(
            (
                config.max_len + 1,
                config.num_attention_heads,
                config.max_len,
                config.max_len,
            ),
            dtype=torch.int,
        )
       
        if ids is None:
            for k in range(config.max_len + 1):
                self.ids[k, :, :k, :k] = self.compute_ids(transpose=False, max_len=k)
                self.ids_t[k, :, :k, :k] = self.compute_ids(transpose=True, max_len=k)
        else:
            self.ids = ids
            self.ids_t = ids_t

        self.scale_mult = torch.tensor(self.attention_head_size, dtype=torch.float)
        self.selector = Selector()

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = nobuco.shape(x)[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, nobuco.shape(x)[1], nobuco.shape(x)[-1])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        ids=None,
        ids_t=None
    ):
        """
        Call the module
        Args:
            hidden_states (`torch.FloatTensor`):
                Input states to the module usually the output from previous layer, it will be the Q,K and V in
                *Attention(Q,K,V)*
            attention_mask (`torch.ByteTensor`):
                An attention mask matrix of shape [*B*, *N*, *N*] where *B* is the batch size, *N* is the maximum
                sequence length in which element [i,j] = *1* means the *i* th token in the input can attend to the *j*
                th token.
            output_attentions (`bool`, optional):
                Whether return the attention matrix.
            query_states (`torch.FloatTensor`, optional):
                The *Q* state in *Attention(Q,K,V)*.
            relative_pos (`torch.LongTensor`):
                The relative position encoding between the tokens in the sequence. It's of shape [*B*, *N*, *N*] with
                values ranging in [*-max_relative_positions*, *max_relative_positions*].
            rel_embeddings (`torch.FloatTensor`):
                The embedding of relative distances. It's a tensor of shape [\\(2 \\times
                \\text{max_relative_positions}\\), *hidden_size*].
        """
        if query_states is None:
            query_states = hidden_states
        query_layer = self.transpose_for_scores(
            self.query_proj(query_states), self.num_attention_heads
        )
        key_layer = self.transpose_for_scores(
            self.key_proj(hidden_states), self.num_attention_heads
        )
        value_layer = self.transpose_for_scores(
            self.value_proj(hidden_states), self.num_attention_heads
        )

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1

        scale = torch.sqrt(self.scale_mult * scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)) / scale

        if self.relative_attention:
            rel_att = self.efficient_disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor, ids=ids, ids_t=ids_t
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        attention_scores = attention_scores.view(
            -1,
            self.num_attention_heads,
            nobuco.shape(attention_scores)[-2],
            nobuco.shape(attention_scores)[-1],
        )

        # bsz x height x length x dimension
        attention_probs = torch.softmax(attention_scores, -1)

        context_layer = torch.bmm(
            attention_probs.view(
                -1, nobuco.shape(attention_probs)[-2], nobuco.shape(attention_probs)[-1]
            ),
            value_layer,
        )
        
        context_layer = (
            context_layer.view(
                -1,
                self.num_attention_heads,
                nobuco.shape(context_layer)[-2],
                nobuco.shape(context_layer)[-1],
            )
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = nobuco.shape(context_layer)[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def compute_ids(self, transpose=True, max_len=50):
        if transpose:
            ids = (
                np.arange(max_len)[None]
                - np.arange(max_len)[:, None]
                + self.position_buckets
            )
        else:
            ids = (
                np.arange(max_len)[:, None]
                - np.arange(max_len)[None]
                + self.position_buckets
            )

        ids += np.arange(0, max_len)[:, None] * self.position_buckets * 2
        ids = ids[None]

        # REPEAT NUM_ATT_HEADS
        ids = torch.from_numpy(ids)
        ids = ids.expand(self.num_attention_heads, -1, -1).contiguous()
        ids += torch.from_numpy(
            np.arange(nobuco.shape(ids)[0]) * self.position_buckets * 2 * max_len
        ).view(-1, 1, 1)

        return ids

    def my_gather(self, c2p_att, dim=2, transpose=False):
        bs, sz, n_fts = nobuco.shape(c2p_att)

        if transpose:
            ids = (
                self.ids_t[sz, :, : sz, : sz]
                .contiguous()
                .view(-1)
            )
        else:
            ids = (
                self.ids[sz, :, : sz, : sz]
                .contiguous()
                .view(-1)
            )
            
        print(ids)
        print(ids.dtype)
        print(ids.size(), ids.sum())

        c2p_att = c2p_att.view(-1)
        
        y = self.selector(c2p_att, ids).view(bs, sz, sz)
#         y = c2p_att[ids].view(bs, sz, sz)
        return y

    def efficient_disentangled_attention_bias(
        self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor, ids=None, ids_t=None
    ):
        att_span = self.pos_ebd_size
        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)

        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            )
            pos_key_layer = self.transpose_for_scores(
                self.key_proj(rel_embeddings), self.num_attention_heads
            )
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                )
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                )
        score = 0

        if "c2p" in self.pos_att_type:  # content->position
            scale = torch.sqrt(self.scale_mult * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            
            if ids is None:
                c2p_att = self.my_gather(c2p_att, dim=2)
            else:
                bs, sz, n_fts = nobuco.shape(c2p_att)
                c2p_att = self.selector(c2p_att.view(-1), ids).view(bs, sz, sz)
                
            score += c2p_att / scale

        if "p2c" in self.pos_att_type:  # position->content
            scale = torch.sqrt(self.scale_mult * scale_factor)
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            
            if ids_t is None:
                p2c_att = self.my_gather(p2c_att, dim=2, transpose=True).transpose(-1, -2)
            else:
                bs, sz, n_fts = nobuco.shape(p2c_att)
                p2c_att = self.selector(p2c_att.view(-1), ids_t).view(bs, sz, sz).transpose(-1, -2)

            score += p2c_att / scale

        return score

    
class Selector(nn.Module):
    def forward(self, x, ids):
#         print(x.dtype, ids.dtype)
        return torch.gather(x, 0, ids.long())
#         return x[ids]
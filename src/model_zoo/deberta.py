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
""" PyTorch DeBERTa-v2 model."""

import numpy as np
from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

# from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder


from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import softmax_backward_data
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers.models.deberta_v2.configuration_deberta_v2 import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import ContextPooler, XSoftmax, DropoutContext, XDropout, StableDropout

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "DebertaV2Config"
_CHECKPOINT_FOR_DOC = "microsoft/deberta-v2-xlarge"
_QA_TARGET_START_INDEX = 2
_QA_TARGET_END_INDEX = 9

DEBERTA_V2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/deberta-v2-xlarge",
    "microsoft/deberta-v2-xxlarge",
    "microsoft/deberta-v2-xlarge-mnli",
    "microsoft/deberta-v2-xxlarge-mnli",
]


# Copied from transformers.models.deberta.modeling_deberta.get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        dropout = local_context.dropout
        dropout *= local_context.scale
        mask = local_context.mask if local_context.reuse_mask else None

    if dropout > 0 and mask is None:
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).bool()  # .to(torch.bool)

    if isinstance(local_context, DropoutContext):
        if local_context.mask is None:
            local_context.mask = mask

    return mask, dropout


# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaLayerNorm->LayerNorm
class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
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
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
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
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
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
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            return layer_output


class ConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        kernel_size = getattr(config, "conv_kernel_size", 3)
        groups = getattr(config, "conv_groups", 1)
        self.conv_act = getattr(config, "conv_act", "tanh")
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups
        )
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        out = ACT2FN[self.conv_act](self.dropout(out))

        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input)  # .to(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask  # .to(output.dtype)
            output_states = output * input_mask

        return output_states


class DebertaV2Encoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()

        self.layer = nn.ModuleList([DebertaV2Layer(config) for _ in range(config.num_hidden_layers)])
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

        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False

    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask

    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
        return relative_pos

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
    ):
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            input_mask = (attention_mask.sum(-2) > 0).byte()
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = None  # self.get_rel_pos(hidden_states, query_states, relative_pos)

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

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                output_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                output_states, att_m = output_states

            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

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
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )


def make_log_bucket_position(relative_pos, bucket_size, max_position):
    sign = torch.where(relative_pos > 0, torch.tensor(1.), torch.tensor(-1.))  # torch.tensor(1, dtype=torch.int64), torch.tensor(-1, dtype=torch.int64))
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).float(),
        torch.sqrt(torch.pow(relative_pos, 2.)),
    )
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    )
    bucket_pos = torch.where(abs_pos <= mid, relative_pos, log_pos * sign)
    return bucket_pos


IDS = torch.tensor(np.arange(1000).astype(np.float32))

def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1, device=None):
    """
    Build relative position according to the query and key
    We assume the absolute position of query \\(P_q\\) is range from (0, query_size) and the absolute position of key
    \\(P_k\\) is range from (0, key_size), The relative positions from query to key is \\(R_{q \\rightarrow k} = P_q -
    P_k\\)
    Args:
        query_size (int): the length of query
        key_size (int): the length of key
        bucket_size (int): the size of position bucket
        max_position (int): the maximum allowed absolute position
        device (`torch.device`): the device on which tensors will be created.
    Return:
        `torch.LongTensor`: A tensor with shape [1, query_size, key_size]
    """
    q_ids = IDS[:query_size].to(device)
    k_ids = IDS[:key_size].to(device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
#     rel_pos_ids = rel_pos_ids.int()  # .long()  # to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.c2p_dynamic_expand
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.p2c_dynamic_expand
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.pos_dynamic_expand
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module
    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]
    """

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        _attention_head_size = config.hidden_size // config.num_attention_heads
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            self.pos_dropout = StableDropout(config.hidden_dropout_prob)

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = StableDropout(config.attention_probs_dropout_prob)
        
#         self.rel_pos =  torch.from_numpy(np.load('rel_pos.npy').astype(np.int32))
        
        sz = 50 * self.num_attention_heads  # max_len * n_heads
        self.eye = np.ravel_multi_index(np.where(np.eye(sz)), (sz, sz))

        self.ids = self.compute_ids(transpose=False, max_len=config.max_len)
        self.ids_t = self.compute_ids(transpose=True, max_len=config.max_len)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
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
        query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)

        rel_att = None
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = torch.sqrt(torch.tensor(query_layer.size(-1), dtype=torch.float) * scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2)) / scale  # .to(dtype=query_layer.dtype)

        if self.relative_attention:
            rel_embeddings = self.pos_dropout(rel_embeddings)
            rel_att = self.efficient_disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )

        if rel_att is not None:
            attention_scores = attention_scores + rel_att

        attention_scores = attention_scores.view(
            -1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1)
        )

        # bsz x height x length x dimension
        attention_probs = XSoftmax.apply(attention_scores, attention_mask, -1)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.bmm(
            attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer
        )
        context_layer = (
            context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1))
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(new_context_layer_shape)
        if output_attentions:
            return (context_layer, attention_probs)
        else:
            return context_layer

    def compute_ids(self, transpose=True, max_len=50):
        if transpose:
            ids = np.arange(max_len)[None] - np.arange(max_len)[:, None] + self.position_buckets
        else:
            ids = np.arange(max_len)[:, None] - np.arange(max_len)[None] + self.position_buckets

#         print(self.position_buckets)
        ids += np.arange(0, max_len)[:, None] * self.position_buckets * 2
        ids = ids[None]
        
        # REPEAT NUM_ATT_HEADS
        ids = torch.from_numpy(ids)
        ids = ids.expand(self.num_attention_heads, -1, -1).contiguous()
        ids += torch.from_numpy(np.arange(ids.size(0)) * self.position_buckets * 2 * max_len).view(-1, 1, 1)
        
        return ids        
        
    def my_gather(self, c2p_att, dim=2, transpose=False):
        bs, sz, n_fts = c2p_att.size()

        if transpose:
            ids = self.ids_t.view(-1)
        else:
            ids = self.ids.view(-1)

        c2p_att = c2p_att.view(-1)

        y = c2p_att[ids].view(bs, sz, sz)
        return y

    def efficient_disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        att_span = self.pos_ebd_size
        rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)
        
#         print(self.share_att_key)
        if self.share_att_key:
#             print(query_layer.size())
            pos_query_layer = self.transpose_for_scores(
                self.query_proj(rel_embeddings), self.num_attention_heads
            )  # .repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
            pos_key_layer = self.transpose_for_scores(
                self.key_proj(rel_embeddings), self.num_attention_heads
            )  # .repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        else:
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(
                    self.pos_key_proj(rel_embeddings), self.num_attention_heads
                )#.repeat(
#                     query_layer.size(0) // self.num_attention_heads, 1, 1
#                 )  # .split(self.all_head_size, dim=-1)
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(
                    self.pos_query_proj(rel_embeddings), self.num_attention_heads
                ) # .repeat(
#                     query_layer.size(0) // self.num_attention_heads, 1, 1
#                 )  # .split(self.all_head_size, dim=-1)
                
        score = 0

        if "c2p" in self.pos_att_type:  # content->position
            scale = torch.sqrt(torch.tensor(pos_key_layer.size(-1), dtype=torch.float) * scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))

            c2p_att = self.my_gather(c2p_att, dim=2)
            score += c2p_att / scale
        
        # position->content
        if "p2c" in self.pos_att_type:
            scale = torch.sqrt(torch.tensor(pos_query_layer.size(-1), dtype=torch.float) * scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                raise NotImplementedError

            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = self.my_gather(p2c_att, dim=2, transpose=True).transpose(-1, -2)
            score += p2c_att / scale

        return score
        
#     def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
#         if True: # self.rel_pos is None:
#             if relative_pos is None:
#                 q = query_layer.size(-2)
#                 relative_pos = build_relative_position(
#                     q,
#                     key_layer.size(-2),
#                     bucket_size=self.position_buckets,
#                     max_position=self.max_relative_positions,
#                     device=query_layer.device,
#                 )
#             if relative_pos.dim() == 2:
#                 relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
#             elif relative_pos.dim() == 3:
#                 relative_pos = relative_pos.unsqueeze(1)
#             # bsz x height x query x key
#             elif relative_pos.dim() != 4:
#                 raise ValueError(f"Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}")

#             relative_pos = relative_pos.long().to(query_layer.device)

#     #         np.save("rel_pos.npy", relative_pos.detach().cpu().numpy())
#         else:
#             relative_pos = self.rel_pos.to(query_layer.device).long()

#         att_span = self.pos_ebd_size
#         rel_embeddings = rel_embeddings[0 : att_span * 2, :].unsqueeze(0)

#         if self.share_att_key:
#             pos_query_layer = self.transpose_for_scores(
#                 self.query_proj(rel_embeddings), self.num_attention_heads
#             ).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
#             pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(
#                 query_layer.size(0) // self.num_attention_heads, 1, 1
#             )
#         else:
#             if "c2p" in self.pos_att_type:
#                 pos_key_layer = self.transpose_for_scores(
#                     self.pos_key_proj(rel_embeddings), self.num_attention_heads
#                 ).repeat(
#                     query_layer.size(0) // self.num_attention_heads, 1, 1
#                 )  # .split(self.all_head_size, dim=-1)
#             if "p2c" in self.pos_att_type:
#                 pos_query_layer = self.transpose_for_scores(
#                     self.pos_query_proj(rel_embeddings), self.num_attention_heads
#                 ).repeat(
#                     query_layer.size(0) // self.num_attention_heads, 1, 1
#                 )  # .split(self.all_head_size, dim=-1)

#         score = 0
#         # content->position
                
#         if "c2p" in self.pos_att_type:
#             scale = torch.sqrt(torch.tensor(pos_key_layer.size(-1), dtype=torch.float) * scale_factor)
#             c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
#             c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
            
#             bs, sz, n_fts = c2p_att.size()
            
# #             print(c2p_att.size(), c2p_pos.size())
# #             print(c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]).size())
# #             print(c2p_pos.squeeze(0))
# #             c2p_att_ = torch.gather(
# #                 c2p_att,
# #                 dim=2,
# #                 index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]),
# #             ) 

#             c2p_att = self.my_gather(c2p_att, c2p_pos, dim=2)
            
# #             print((c2p_att == c2p_att_).all())

#             score += c2p_att / scale  # .to(dtype=c2p_att.dtype)

#         # position->content
#         if "p2c" in self.pos_att_type:
#             scale = torch.sqrt(torch.tensor(pos_query_layer.size(-1), dtype=torch.float) * scale_factor)
#             if key_layer.size(-2) != query_layer.size(-2):
#                 r_pos = build_relative_position(
#                     key_layer.size(-2),
#                     key_layer.size(-2),
#                     bucket_size=self.position_buckets,
#                     max_position=self.max_relative_positions,
#                     device=query_layer.device,
#                 )
#                 r_pos = r_pos.unsqueeze(0)
#             else:
#                 r_pos = relative_pos

#             p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
#             p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))

# #             p2c_att = torch.gather(
# #                 p2c_att,
# #                 dim=2,
# #                 index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]),
# #             ).transpose(-1, -2)

#             p2c_att = self.my_gather(p2c_att, p2c_pos, dim=2, transpose=True).transpose(-1, -2)

# #             print((p2c_att == p2c_att_).all())

#             score += p2c_att / scale  # .to(dtype=p2c_att.dtype)

#         return score

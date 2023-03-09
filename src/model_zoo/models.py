import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Encoder

# from model_zoo.gem import GeM
from utils.torch import load_model_weights


def define_model(
    name,
    embed_dim=256,
    transfo_dim=768,
    transfo_heads=1,
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

    if name == "bi_transfo":
        model = SignBiTransformer(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name == "multi_transfo":
        model = SignMultiTransformer(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name == "transfo_deberta":
        model = SignTransformerDeberta(
            embed_dim=embed_dim,
            transfo_dim=transfo_dim,
            transfo_heads=transfo_heads,
            num_classes=num_classes,
            drop_rate=drop_rate,
        )
    elif name == "bert_deberta":
        model = SignBertDeberta(
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


class SignBertDeberta(nn.Module):
    """
    Model with an attention mechanism.
    """
    def update_config(self, config):
        pass
        
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
        self.landmark_embed = nn.Embedding(127, embed_dim, padding_idx=0)
        
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)
        
        self.pos_dense = nn.Linear(3, embed_dim)

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

        bs, n_frames, n_landmarks, _ = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

        fts = self.landmark_transformer(fts).last_hidden_state
        fts = fts.view(bs, n_frames, n_landmarks, -1)
        fts = fts.mean(2)

#         hidden_states = self.frame_transformer(fts, x['mask'][:, :, 0]).hidden_states
#         fts = torch.cat(hidden_states[::-1], -1)
        fts = self.frame_transformer(fts, x['mask'][:, :, 0]).last_hidden_state
        
#         print(fts.size())
        fts = fts.view(bs, n_frames, -1)
        fts = fts.mean(1)

#         print(fts.size())
        
        logits = self.logits(fts)

        return logits, torch.zeros(1)


class SignTransformerDeberta(nn.Module):
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
        self.landmark_embed = nn.Embedding(127, embed_dim, padding_idx=0)
        
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)
        
        self.pos_dense = nn.Linear(3, embed_dim)
        nb_fts = embed_dim * 3
        
        self.landmark_transformer = nn.TransformerEncoderLayer(
            nb_fts,
            nhead=transfo_heads,
            dim_feedforward=transfo_dim,
            dropout=drop_rate,
            batch_first=True
        )
        
        name = "microsoft/deberta-v3-base"
        config = AutoConfig.from_pretrained(name, output_hidden_states=True)
        config.hidden_size = transfo_dim
        config.intermediate_size = transfo_dim * 4
        config.num_hidden_layers = 1
        config.num_attention_heads = transfo_heads
        config.attention_probs_dropout_prob = drop_rate
        config.hidden_dropout_prob = drop_rate
        config.max_relative_positions = 100

        self.frame_transformer = DebertaV2Encoder(config)

        self.logits = nn.Linear(nb_fts , num_classes)

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

        bs, n_frames, n_landmarks, nb_fts = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

        fts = self.landmark_transformer(fts)  # , src_mask=mask)
        fts = fts.view(bs, n_frames, n_landmarks, -1)
        fts = fts.mean(2)
        
#         hidden_states = self.frame_transformer(fts, x['mask'][:, :, 0]).hidden_states
#         fts = torch.cat(hidden_states[::-1], -1)
        fts = self.frame_transformer(fts, x['mask'][:, :, 0]).last_hidden_state
        
#         print(fts.size())
        fts = fts.view(bs, n_frames, -1)
        fts = fts.mean(1)

#         print(fts.size())
        
        logits = self.logits(fts)

        return logits, torch.zeros(1)


class SignBiTransformer(nn.Module):
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
        self.landmark_embed = nn.Embedding(127, embed_dim, padding_idx=0)
        
        self.type_norm = nn.LayerNorm(embed_dim)
        self.landmark_norm = nn.LayerNorm(embed_dim)
        
        self.pos_dense = nn.Linear(3, embed_dim)
        nb_fts = embed_dim * 3
        
        self.landmark_transformer = nn.TransformerEncoderLayer(
            nb_fts,
            nhead=transfo_heads,
            dim_feedforward=transfo_dim,
            dropout=drop_rate,
            batch_first=True
        )

        self.frame_transformer = nn.TransformerEncoderLayer(
            nb_fts,
            nhead=transfo_heads,
            dim_feedforward=transfo_dim,
            dropout=drop_rate,
            batch_first=True
        )

        self.logits = nn.Linear(nb_fts , num_classes)

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

        bs, n_frames, n_landmarks, nb_fts = fts.size()
        fts = fts.view(bs * n_frames, n_landmarks, -1)

#         mask = x['mask'].view(bs * n_frames, 1, n_landmarks, 1).repeat(1, self.transfo_heads, 1, n_landmarks)
#         mask = mask.view(bs * n_frames * self.transfo_heads, n_landmarks, n_landmarks)
#         mask = None

        fts = self.landmark_transformer(fts)  # , src_mask=mask)
        fts = fts.view(bs, n_frames, n_landmarks, -1)
        fts = fts.mean(2)

#         mask = (x['mask'].unsqueeze(-1)).repeat(1, 1, 1, fts.size(-1))
#         fts = fts.masked_fill_(mask, 0)
        
        fts = self.frame_transformer(fts)
        fts = fts.view(bs, n_frames, -1)
        fts = fts.mean(1)

#         print(fts.size())
        
        logits = self.logits(fts)

        return logits, torch.zeros(1)

    
class SignMultiTransformer(nn.Module):
    """
    Model with an attention mechanism.
    """
    def __init__(
        self,
        embed_dim=256,
        transfo_dim=768,
        transfo_heads=1,
        num_classes=250,
        n_transfos=2,
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
        self.n_transfos = n_transfos

        self.type_embed = nn.Embedding(12, embed_dim, padding_idx=0)
        self.landmark_embed = nn.Embedding(127, embed_dim, padding_idx=0)
        self.pos_dense = nn.Linear(3, embed_dim)
        nb_fts = embed_dim * 3
        
        self.transformers_1 = []
        self.transformers_2 = []
        for _ in range(n_transfos):
            self.transformers_1.append(
                nn.TransformerEncoderLayer(
                    nb_fts,
                    nhead=transfo_heads,
                    dim_feedforward=transfo_dim,
                    dropout=drop_rate,
                    batch_first=True
                )
            )
            self.transformers_2.append(
                nn.TransformerEncoderLayer(
                    nb_fts,
                    nhead=transfo_heads,
                    dim_feedforward=transfo_dim,
                    dropout=drop_rate,
                    batch_first=True
                )
            )
            
        self.transformers_1 = nn.ModuleList(self.transformers_1)
        self.transformers_2 = nn.ModuleList(self.transformers_2)

        self.logits = nn.Linear(nb_fts , num_classes)

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
        x_type = self.type_embed(x['type'])
        x_landmark = self.landmark_embed(x['landmark'])
        x_pos = self.pos_dense(torch.stack([x['x'], x['y'], x['z']], -1))
       
        fts = torch.cat([x_type, x_landmark, x_pos], -1)

        bs, n_frames, n_landmarks, nb_fts = fts.size()
        
        for i in range(self.n_transfos):
            fts = fts.view(bs * n_frames, n_landmarks, -1)
            fts = self.transformers_1[i](fts)  # , src_mask=mask)
            fts = fts.view(bs, n_frames, n_landmarks, -1)

            fts = fts.transpose(1, 2).contiguous().view(bs * n_landmarks, n_frames, -1)

            fts = self.transformers_2[i](fts)
            fts = fts.view(bs, n_landmarks, n_frames, -1).transpose(1, 2).contiguous()
        
        fts = fts.mean(1).mean(1)
        
        logits = self.logits(fts)

        return logits, torch.zeros(1)

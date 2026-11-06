import math

import torch
import torch.nn as nn

from SANE.datasets.dataset_auxiliaries import tokenize_checkpoint
from SANE.git_re_basin.git_re_basin import apply_permutation, weight_matching
from SANE.models.def_AE import PositionEmbs
from SANE.models.def_transformer import TransformerEncoder

SANE_ENCODER_PREFIXES = ("tokenizer.", "pe.", "transformer_encoder.")


def prepare_tokens(checkpoint, preprocessing, permutation_spec):
    """Tokens [n_tokens, tokensize] of one raw checkpoint (a state dict), as DatasetTokens makes them.

    preprocessing: the preprocessing.pt of the zoo (reference_checkpoint, layers, tokensize).
    1. Canonical form: permute the checkpoint to match the reference checkpoint of the zoo.
    2. Standardize each layer (weight and bias) with the mean and std of the train split.
    3. Cut the layers into tokens (ignore_bn=True)."""
    permutation = weight_matching(ps=permutation_spec, params_a=preprocessing["reference_checkpoint"], params_b=checkpoint)
    checkpoint = apply_permutation(ps=permutation_spec, perm=permutation, params=checkpoint)
    for weight_key, stats in preprocessing["layers"].items():
        for key in (weight_key, weight_key.replace("weight", "bias")):
            if key in checkpoint:
                checkpoint[key] = (checkpoint[key] - stats["mean"]) / stats["std"]
    tokens, _, _ = tokenize_checkpoint(checkpoint, tokensize=preprocessing["tokensize"], return_mask=True, ignore_bn=True)
    return tokens


class RecallTransformer(nn.Module):
    """Transformer encoder that maps the weight tokens of one model to one output per class.

    The tokenizer, position embeddings and transformer are the same modules as in the SANE
    encoder (def_AE.AE), so a pretrained SANE encoder loads into it (load_sane_encoder).
    A learned CLS token goes in front of the token sequence. Its output goes through a
    layer norm and one linear layer to n_classes outputs. There is no bottleneck.
    """

    def __init__(self, i_dim, d_model, nhead, num_layers, max_positions, n_classes, dropout):
        super().__init__()
        self.tokenizer = nn.Linear(i_dim, d_model)
        self.pe = PositionEmbs(max_positions=max_positions, embedding_dim=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)
        # block_size is only used for a causal mask, so it does not limit the sequence length here
        self.transformer_encoder = TransformerEncoder(
            n_layer=num_layers, n_head=nhead, d_model=d_model, dropout=dropout, bias=False, causal=False
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

        # initialization as in def_AE.AE (GPT-2)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        torch.nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def forward(self, tokens, positions):
        """tokens [batch, n_tokens, i_dim], positions [batch, n_tokens, len(max_positions)] -> [batch, n_classes]"""
        x = self.pe(self.tokenizer(tokens), positions)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.dropout(torch.cat([cls_tokens, x], dim=1))
        x = self.transformer_encoder(x)
        return self.head(x[:, 0])

    def load_sane_encoder(self, sane_state_dict):
        """Load tokenizer, pe and transformer_encoder from the state dict of a SANE AE.

        The CLS token and the head keep their random initialization."""
        encoder_state = {k: v for k, v in sane_state_dict.items() if k.startswith(SANE_ENCODER_PREFIXES)}
        missing, unexpected = self.load_state_dict(encoder_state, strict=False)
        assert not unexpected and all(k == "cls_token" or k.startswith("head.") for k in missing), missing


class ClassTokenRecallTransformer(nn.Module):
    """Transformer encoder with one input token and one output per class.

    Class token c is the last-layer (fc) row of class c: its weights and bias, class_dim values.
    The body tokens (all layers before fc) are optional and go in front, with position
    embeddings as in RecallTransformer. One shared head maps the output at each class token to
    one value. class_index=False: all class tokens get the same learned embedding, so the model
    treats all classes alike. class_index=True: each class gets its own learned embedding.
    """

    def __init__(self, class_dim, n_classes, class_index, d_model, nhead, num_layers, dropout, body_dim=None, max_positions=None):
        super().__init__()
        self.n_classes = n_classes
        self.class_tokenizer = nn.Linear(class_dim, d_model)
        self.class_embedding = nn.Parameter(torch.zeros(1, n_classes if class_index else 1, d_model))
        if body_dim:
            self.body_tokenizer = nn.Linear(body_dim, d_model)
            self.pe = PositionEmbs(max_positions=max_positions, embedding_dim=d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer_encoder = TransformerEncoder(
            n_layer=num_layers, n_head=nhead, d_model=d_model, dropout=dropout, bias=False, causal=False
        )
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        # initialization as in RecallTransformer (GPT-2)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        torch.nn.init.normal_(self.class_embedding, mean=0.0, std=0.02)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                torch.nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * num_layers))

    def forward(self, class_tokens, body_tokens=None, body_positions=None):
        """class_tokens [batch, n_classes, class_dim], body_tokens [batch, n_body, body_dim] -> [batch, n_classes]"""
        x = self.class_tokenizer(class_tokens) + self.class_embedding
        if body_tokens is not None:
            x = torch.cat([self.pe(self.body_tokenizer(body_tokens), body_positions), x], dim=1)
        x = self.transformer_encoder(self.dropout(x))
        return self.head(x[:, -self.n_classes :]).squeeze(-1)

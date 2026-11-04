import math

import torch
import torch.nn as nn

from SANE.models.def_AE import PositionEmbs
from SANE.models.def_transformer import TransformerEncoder

SANE_ENCODER_PREFIXES = ("tokenizer.", "pe.", "transformer_encoder.")


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

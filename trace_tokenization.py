"""
Trace how DatasetTokens.tokenize_checkpoint slices a model into tokens,
starting from a raw flat weight array (the same format as model_export/SHAPES).

Run: .venv/bin/python3 trace_tokenization.py
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import torch

# ---- mock pytorch_port so we can import reconstruct_network helpers ----
sys.modules["pytorch_port"] = MagicMock()
sys.modules["pytorch_port.model"] = MagicMock()
sys.modules["pytorch_port.model"].SmallCNN = MagicMock()

from model_export.reconstruct_network import (
    SHAPES,
    TOTAL_PARAMS,
    _flat_weights_to_state_dict,
)
from SANE.datasets.dataset_auxiliaries import tokenize_checkpoint, tokens_to_checkpoint

# ---- 1. Create a random flat array and convert to PyTorch state dict ----
np.random.seed(42)
flat = np.random.randn(TOTAL_PARAMS).astype(np.float32)
state_dict = _flat_weights_to_state_dict(flat)

print("=== PyTorch state dict shapes ===")
for k, v in state_dict.items():
    print(f"  {k:20s}  shape={tuple(v.shape)}")
# conv1.weight    (16, 1, 3, 3)
# conv1.bias      (16,)
# conv2.weight    (16, 16, 3, 3)
# conv2.bias      (16,)
# conv3.weight    (16, 16, 3, 3)
# conv3.bias      (16,)
# dense.weight    (10, 16)
# dense.bias      (10,)

# ---- 2. Tokenize with tokensize=0 (auto-discover) ----
tokens, mask, pos = tokenize_checkpoint(
    checkpoint=state_dict,
    tokensize=0,          # auto-discover
    return_mask=True,
    ignore_bn=False,
)

print(f"\n=== Tokens info ===")
print(f"  tokensize (auto-discovered) = {tokens.shape[1]}")
print(f"  total tokens               = {tokens.shape[0]}")

# ---- 3. Show per-layer breakdown ----
print(f"\n=== Per-layer token breakdown ===")
layer_names = ["conv1", "conv2", "conv3", "dense"]
# pos columns: [global_token_idx, layer_idx, channel_in_layer]
for lidx, lname in enumerate(layer_names):
    layer_mask = pos[:, 1] == lidx
    n_tokens = layer_mask.sum().item()
    n_channels = pos[layer_mask, 2].max().item() + 1
    token_dim = tokens.shape[1]
    # How many real (non-padded) values per channel?
    w = state_dict[f"{lname}.weight"]
    per_channel = w[0].numel()  # for weight only
    if f"{lname}.bias" in state_dict:
        per_channel += 1
    n_real = n_channels * per_channel
    n_total = n_tokens * token_dim
    n_pad = n_total - n_real
    token_factor = n_tokens // n_channels
    print(f"  {lname}: {n_channels} channels × {token_factor} token(s) each "
          f"= {n_tokens} tokens, "
          f"per-channel size={per_channel}, "
          f"real values={n_real}, padding={n_pad} "
          f"(mask % true = {mask[layer_mask].float().mean().item():.4f})")

# ---- 4. Trace back what's IN a single token for dense.weight ----
print(f"\n=== Anatomy of one dense.weight token ===")
dense_layer_idx = 3
dense_tokens = tokens[pos[:, 1] == dense_layer_idx]
dense_mask_vals = mask[pos[:, 1] == dense_layer_idx]
print(f"  Token shape: {dense_tokens.shape}")  # (10, 145)
print(f"  Original dense.weight: {tuple(state_dict['dense.weight'].shape)}")  # (10, 16)
print(f"  Original dense.bias:   {tuple(state_dict['dense.bias'].shape)}")    # (10,)
print(f"  Token layout: first 16 values = weight row, value 17 = bias, rest padding")
print(f"  Token 0 (channel 0):")
print(f"    weight row: {dense_tokens[0, :16]}")
print(f"    bias:       {dense_tokens[0, 16]:.4f}")
print(f"    padded?     {(~dense_mask_vals[0]).sum().item()} zeros (mask=false)")

# Verify: compare first token's values to original
weight_row_0 = state_dict["dense.weight"][0]
bias_0 = state_dict["dense.bias"][0]
print(f"  Sanity check — weight match: {torch.allclose(dense_tokens[0, :16], weight_row_0)}")
print(f"  Sanity check — bias match:   {torch.allclose(dense_tokens[0, 16], bias_0)}")

# ---- 5. Round-trip: tokens -> checkpoint ----
reconstructed = tokens_to_checkpoint(
    tokens=tokens,
    pos=pos,
    reference_checkpoint=state_dict,
    ignore_bn=False,
)

print(f"\n=== Round-trip fidelity ===")
all_ok = True
for k in state_dict:
    ok = torch.allclose(state_dict[k], reconstructed[k], atol=1e-6)
    if not ok:
        max_diff = (state_dict[k] - reconstructed[k]).abs().max().item()
        print(f"  {k}: MISMATCH (max diff={max_diff:.6e})")
        all_ok = False
    else:
        print(f"  {k}: OK")
print(f"\n  ALL OK: {all_ok}")

# ---- 6. Position tensor anatomy ----
print(f"\n=== Position tensor anatomy ===")
print(f"  Shape: {pos.shape}  (n_tokens, 3)")
print(f"  Columns: [global_token_index, layer_index, channel_in_layer]")
print(f"  First 3 rows:  {pos[:3].tolist()}")
print(f"  Last  3 rows:  {pos[-3:].tolist()}")
print(f"  Unique layers: {pos[:, 1].unique().tolist()}")
print(f"  Per-layer token counts: "
      f"{[(l, (pos[:, 1] == l).sum().item()) for l in range(4)]}")

"""Differentiable flat-weight <-> state-dict <-> tokens pipeline for SmallCNN.

Mirrors the TF-storage-order convention from model_export.reconstruct_network.
All operations are torch-native — no numpy, no .data.copy_, no deepcopy —
so gradients flow end-to-end.
"""
from math import prod
import torch

from model_export.reconstruct_network import SHAPES, TOTAL_PARAMS

# TF-name → (pt_key, forward_transpose_axes)
_TF_TO_PT = [
    ("sequential/conv2d/bias:0",      "conv1.bias",    None),
    ("sequential/conv2d/kernel:0",    "conv1.weight",  (3, 2, 0, 1)),
    ("sequential/conv2d_1/bias:0",    "conv2.bias",    None),
    ("sequential/conv2d_1/kernel:0",  "conv2.weight",  (3, 2, 0, 1)),
    ("sequential/conv2d_2/bias:0",    "conv3.bias",    None),
    ("sequential/conv2d_2/kernel:0",  "conv3.weight",  (3, 2, 0, 1)),
    ("sequential/dense/bias:0",       "dense.bias",    None),
    ("sequential/dense/kernel:0",     "dense.weight",  (1, 0)),
]

_WEIGHT_KEYS = {"conv1.weight", "conv2.weight", "conv3.weight", "dense.weight"}
_BIAS_KEYS   = {"conv1.bias", "conv2.bias", "conv3.bias", "dense.bias"}
_LAYER_ORDER  = ["conv1", "conv2", "conv3", "dense"]


def _build_ref_shapes():
    """Build reference shape dict once for detokenize."""
    sd = {}
    for tf_name, pt_key, transpose in _TF_TO_PT:
        tf_shape = SHAPES[tf_name]
        shape = tf_shape
        if transpose is not None:
            shape = _apply_transpose_to_shape(tf_shape, transpose)
        sd[pt_key] = torch.Size(shape)
    return sd


def _apply_transpose_to_shape(shape, perm):
    return tuple(shape[p] for p in perm)


REF_SHAPES = _build_ref_shapes()


def flat_to_state_dict(flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert flat tensor (TF storage order) → PyTorch state dict.  Grad-safe.

    Args:
        flat: 1-D tensor of length TOTAL_PARAMS.

    Returns:
        dict mapping PT keys to tensors sharing flat's grad graph.
    """
    assert flat.shape == (TOTAL_PARAMS,), f"Expected ({TOTAL_PARAMS},), got {flat.shape}"
    sd = {}
    offset = 0
    for tf_name, pt_key, transpose in _TF_TO_PT:
        tf_shape = SHAPES[tf_name]
        n = 1
        for d in tf_shape:
            n *= d
        chunk = flat[offset : offset + n].reshape(tf_shape)
        if transpose is not None:
            chunk = chunk.permute(*transpose)
        sd[pt_key] = chunk
        offset += n
    return sd


def state_dict_to_flat(sd: dict[str, torch.Tensor]) -> torch.Tensor:
    """PyTorch state dict → flat tensor (TF order).  Grad-safe."""
    pieces = []
    for tf_name, pt_key, transpose in _TF_TO_PT:
        t = sd[pt_key]
        if transpose is not None:
            inv = tuple(torch.argsort(torch.tensor(transpose)).tolist())
            t = t.permute(*inv)
        pieces.append(t.reshape(-1))
    return torch.cat(pieces, dim=0)


def standardize_state_dict(
    sd: dict[str, torch.Tensor],
    stats: dict[str, dict[str, float]],
) -> dict[str, torch.Tensor]:
    """Apply per-layer standardization: (x - mean) / std.  Grad-safe.

    Mirrors ``DatasetTokens.standardize_data_checkpoints``: both the weight
    tensor and its corresponding bias get the same per-layer mean/std.

    Args:
        sd:    PyTorch state dict (from ``flat_to_state_dict``).
        stats: dict mapping ``*.weight`` keys to ``{"mean": float, "std": float}``.
               Typically extracted from a cached ``DatasetTokens.layers``.

    Returns:
        New state dict with standardized tensors (grad graph preserved).
    """
    out = dict(sd)
    for key, st in stats.items():
        if key not in out:
            continue
        mu, sigma = st["mean"], st["std"]
        out[key] = (out[key] - mu) / sigma
        bias_key = key.replace("weight", "bias")
        if bias_key in out:
            out[bias_key] = (out[bias_key] - mu) / sigma
    return out


def tokenize(
    sd: dict[str, torch.Tensor],
    tokensize: int = 0,
    ignore_bn: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """State dict → (tokens, mask, pos).  Grad-safe.

    Mirrors `SANE.datasets.dataset_auxiliaries.tokenize_checkpoint`.
    tokensize=0 means auto-discover from layer sizes.

    Returns:
        tokens:  (N_tokens, tokensize)  float
        mask:    (N_tokens, tokensize)  bool
        pos:     (N_tokens, 3)          int  [global_idx, layer_idx, chan_idx]
    """
    if tokensize <= 0:
        tokensize = _discover_tokensize(sd, ignore_bn)
    tokensize = int(tokensize)

    all_tokens, all_masks, all_pos = [], [], []
    global_idx = 0

    for lidx, lname in enumerate(_LAYER_ORDER):
        wkey, bkey = f"{lname}.weight", f"{lname}.bias"
        if wkey not in sd:
            continue

        w = sd[wkey]
        w = w.reshape(w.shape[0], -1)
        if bkey in sd:
            w = torch.cat([w, sd[bkey].unsqueeze(1)], dim=1)

        C_out, per_chan = w.shape[0], w.shape[1]
        token_factor = (per_chan + tokensize - 1) // tokensize
        padded_len = token_factor * tokensize

        if per_chan < padded_len:
            w_padded = torch.zeros(C_out, padded_len, device=w.device, dtype=w.dtype)
            w_padded[:, :per_chan] = w
            mask_layer = torch.zeros(C_out, padded_len, device=w.device, dtype=torch.bool)
            mask_layer[:, :per_chan] = True
        else:
            w_padded = w
            mask_layer = torch.ones(C_out, padded_len, device=w.device, dtype=torch.bool)

        all_tokens.append(w_padded.reshape(-1, tokensize))
        all_masks.append(mask_layer.reshape(-1, tokensize))
        for c in range(C_out):
            for _ in range(token_factor):
                all_pos.append((global_idx, lidx, c))
                global_idx += 1

    tokens = torch.cat(all_tokens, dim=0)
    mask = torch.cat(all_masks, dim=0)
    pos = torch.tensor(all_pos, dtype=torch.int)
    return tokens, mask, pos


def detokenize(
    tokens: torch.Tensor,
    pos: torch.Tensor,
    ignore_bn: bool = False,
) -> dict[str, torch.Tensor]:
    """Tokens → state dict.  Grad-safe.  Uses REF_SHAPES for layer topology.

    Args:
        tokens:  (N, tokensize)  float  (may carry grad graph)
        pos:     (N, 3)          int

    Returns:
        State dict whose tensors depend on tokens (grad flows through
        index_select + reshape + slice).
    """
    sd = {}
    for lidx, lname in enumerate(_LAYER_ORDER):
        wkey, bkey = f"{lname}.weight", f"{lname}.bias"
        if wkey not in REF_SHAPES:
            continue

        indices = torch.where(pos[:, 1] == lidx)[0]
        layer_tokens = torch.index_select(tokens, dim=0, index=indices)

        w_shape = REF_SHAPES[wkey]
        C_out = w_shape[0]
        per_chan_weight = w_shape.numel() // C_out
        per_chan = per_chan_weight + (1 if bkey in REF_SHAPES else 0)

        flat = layer_tokens.reshape(C_out, -1)
        content = flat[:, :per_chan]

        sd[wkey] = content[:, :per_chan_weight].reshape(w_shape)
        if bkey in REF_SHAPES:
            sd[bkey] = content[:, per_chan_weight]

    return sd


def _discover_tokensize(sd, ignore_bn):
    tsize = 0
    for key in sd:
        if ignore_bn and ("bn" in key or "downsample.1" in key):
            continue
        if not ("weight" in key or "running_mean" in key or "running_var" in key):
            continue
        w = sd[key]
        per_chan = w.numel() // w.shape[0]
        if key.replace("weight", "bias") in sd:
            per_chan += 1
        tsize = max(tsize, per_chan)
    return tsize

"""SANE as a meta-network for the `wse` unlearning pipeline.

`wse` knows one contract and nothing about who implements it:

    n_classes: int
    __call__(params: {name: (B, *shape)}) -> (B, n_classes)    differentiable
    prepare(shapes: {name: torch.Size}, device) -> None        optional
    embed(params) -> (B, latent)                               optional

This module adapts SANE's three native artifacts to that contract, so nothing in `wse` imports
SANE:

    AE trial dir (params.json + checkpoint_*/state.pt)  ->  the encoder
    ridge head B: [lat_dim + 1, n_classes]              ->  one nn.Linear, y = [z, 1] @ B
    preprocessing params                                ->  standardization, and the canonical
                                                            frame when one is named

Load it by spec string, which is why the manifest exists:

    uv run wse check-lens 'sane_wse:from_manifest(path=/path/to/manifest.yaml)' --zoo <zoo>

`prepare()` builds a static token plan from the zoo's reference *shapes*. Tokenization depends on
shapes alone, never on values, so a whole batch is tokenized by one gather instead of the per-model
Python loop in `model_export.model_wapper.SANEModelWrapper`.
"""

from __future__ import annotations

import json
from math import prod
from pathlib import Path

import torch
import yaml
from torch import Tensor, nn


def from_manifest(path: str | Path) -> "SANELens":
    """Build the lens from a YAML manifest. Every path in it is relative to the manifest itself.

    Keys:
        ae_trial_dir          SANE pretraining trial dir (params.json + checkpoint_*/state.pt)
        prediction_head       .pt holding "B": [lat_dim + 1, n_classes]
        standardization       {"<layer>.weight": {mean, std}}, the stats the AE was trained on
        ignore_bn             must match preprocessing (default True)
        permutation_spec      name in SANE.git_re_basin.git_re_basin; null leaves models un-aligned
        reference_checkpoint  .pt state dict to align to; required with a permutation_spec
        clamp_min             clamp predictions from below, or null for the raw ridge output

    sane_wse_from_cache.py writes the standardization block and the reference checkpoint out of the
    cached DatasetTokens the head was fit on, which is the only record of both.
    """
    manifest_path = Path(path).expanduser().resolve()
    manifest = yaml.safe_load(manifest_path.read_text())
    root = manifest_path.parent

    encoder, config = _load_encoder(root / manifest["ae_trial_dir"])
    reference_checkpoint, permutation_spec = _load_canonical_frame(root, manifest)
    return SANELens(
        encoder=encoder,
        head=_load_head(root / manifest["prediction_head"]),
        tokensize=config["ae:i_dim"],
        ignore_bn=bool(manifest.get("ignore_bn", True)),
        standardization=manifest["standardization"],
        reference_checkpoint=reference_checkpoint,
        permutation_spec=permutation_spec,
        clamp_min=manifest.get("clamp_min"),
    )


class SANELens(nn.Module):
    """`{name: (B, *shape)} -> (B, n_classes)`: standardize, tokenize, encode, ridge head."""

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module,
        tokensize: int,
        ignore_bn: bool,
        standardization: dict[str, dict[str, float]],
        reference_checkpoint: dict[str, Tensor] | None = None,
        permutation_spec=None,
        clamp_min: float | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.n_classes = next(m.out_features for m in reversed(list(head.modules())) if isinstance(m, nn.Linear))
        self.tokensize = int(tokensize)
        self.ignore_bn = ignore_bn
        self.standardization = standardization
        self.reference_checkpoint = reference_checkpoint
        self.permutation_spec = permutation_spec
        self.clamp_min = clamp_min
        self.order: tuple[str, ...] = ()
        self.n_tokens = 0

    # ------------------------------------------------------------------ contract
    def prepare(self, shapes: dict[str, torch.Size], device: torch.device) -> None:
        """Derive the static token plan, once, from shapes alone.

        The plan is a gather index over the concatenated state dict plus one appended zero column,
        which is where every padded token slot reads from. Standardization is folded into a
        per-element mean/std vector over the same layout, because it is affine and per layer.
        """
        self.order = tuple(shapes)
        sizes = {name: prod(shape) for name, shape in shapes.items()}
        width = sum(sizes.values())
        assert width < 2**24, f"{width} parameters exceeds the exact integer range of float32"

        starts, offset = {}, 0
        index_checkpoint = {}
        for name, shape in shapes.items():
            starts[name] = offset
            index_checkpoint[name] = torch.arange(offset, offset + sizes[name], dtype=torch.float64).reshape(shape)
            offset += sizes[name]

        tokens, masks, pos = _plan_from_checkpoint(
            index_checkpoint, tokensize=self.tokensize, ignore_bn=self.ignore_bn
        )
        self.n_tokens = tokens.shape[0]
        self.register_buffer("plan", torch.where(masks.flatten(), tokens.flatten().to(torch.long), width))
        self.register_buffer("pos", pos.to(torch.int))

        mean, std = torch.zeros(width), torch.ones(width)
        for key, stat in self.standardization.items():
            for name in (key, key.replace("weight", "bias")):
                if name in sizes:
                    mean[starts[name] : starts[name] + sizes[name]] = stat["mean"]
                    std[starts[name] : starts[name] + sizes[name]] = stat["std"]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.to(device)

    def forward(self, params: dict[str, Tensor]) -> Tensor:
        """Predicted per-class accuracy, `(B, n_classes)`.

        The ridge head is a plain linear map, so a class predicted near zero accuracy can come out
        slightly negative. `clamp_min` is off by default: clamping kills the gradient of exactly
        the class being unlearned. Use `--stopping acc_pred` with an absolute threshold instead.
        """
        prediction = self.head(self.embed(params))
        return prediction if self.clamp_min is None else prediction.clamp(min=self.clamp_min)

    def embed(self, params: dict[str, Tensor]) -> Tensor:
        """The SANE embedding of each model, `(B, lat_dim)`."""
        tokens = self._tokens(params)
        return self.encoder.forward_embeddings(tokens, self.pos.expand(tokens.shape[0], -1, -1))

    # ------------------------------------------------------------------ internals
    def _tokens(self, params: dict[str, Tensor]) -> Tensor:
        """`{name: (B, *shape)} -> (B, n_tokens, tokensize)`, in one gather."""
        if self.permutation_spec is not None:
            params = self._canonicalize(params)
        batch = next(iter(params.values())).shape[0]
        source = torch.cat([params[name].reshape(batch, -1) for name in self.order], dim=1)
        source = (source - self.mean) / self.std
        source = torch.cat([source, source.new_zeros(batch, 1)], dim=1)
        return source[:, self.plan].view(batch, self.n_tokens, self.tokensize)

    def _canonicalize(self, params: dict[str, Tensor]) -> dict[str, Tensor]:
        """Align every model to the reference frame, as `map_to_canonical=True` does at training.

        `weight_matching` is a discrete per-model search, so this is a Python loop and cannot be
        batched. It exists to measure whether canonicalization matters, not to run a sweep.
        """
        from SANE.git_re_basin.git_re_basin import apply_permutation, weight_matching

        batch = next(iter(params.values())).shape[0]
        aligned = []
        for index in range(batch):
            model = {name: tensor[index] for name, tensor in params.items()}
            permutation = weight_matching(
                self.permutation_spec,
                self.reference_checkpoint,
                {name: tensor.detach() for name, tensor in model.items()},
            )
            aligned.append(apply_permutation(self.permutation_spec, permutation, model))
        return {name: torch.stack([model[name] for model in aligned]) for name in self.order}


# ---------------------------------------------------------------------- token plan
def _plan_from_checkpoint(
    checkpoint: dict[str, Tensor],
    tokensize: int,
    ignore_bn: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    tokens, masks, positions = [], [], []
    tokensize_int = int(tokensize)
    layer_index = 0

    for key in checkpoint:
        if ignore_bn and ("bn" in key or "downsample.1" in key or "batchnorm" in key):
            continue
        if key == "num_batches_tracked":
            continue
        if not ("weight" in key or "running_mean" in key or "running_var" in key):
            continue

        weight = checkpoint[key]
        original = weight.reshape(weight.shape[0], -1)

        if "weight" in key and key.replace("weight", "bias") in checkpoint:
            bias = checkpoint[key.replace("weight", "bias")]
            bias = bias.reshape(-1, 1).expand(original.shape[0], 1)
            original = torch.cat([original, bias], dim=1)

        out_channels = original.shape[0]
        cols = original.shape[1]
        token_factor = (cols + tokensize_int - 1) // tokensize_int

        padded = torch.zeros(out_channels, token_factor * tokensize_int)
        padded[:, :cols] = original
        mask_row = torch.zeros(out_channels, token_factor * tokensize_int)
        mask_row[:, :cols] = 1.0

        tokens.append(padded.view(-1, tokensize_int))
        masks.append(mask_row.view(-1, tokensize_int).to(torch.bool))
        for channel in range(out_channels):
            for fragment in range(token_factor):
                positions.append([layer_index, channel])
        layer_index += 1

    tokens = torch.cat(tokens, dim=0)
    masks = torch.cat(masks, dim=0)
    positions = [(ndx, idx, jdx) for ndx, (idx, jdx) in enumerate(positions)]
    positions = torch.tensor(positions)
    positions = positions.to(torch.int16 if positions.max() <= 32767 else torch.int)
    return tokens, masks, positions


def _load_encoder(trial_dir: Path) -> tuple[nn.Module, dict]:
    from SANE.models.def_AE import AE

    config = json.loads((trial_dir / "params.json").read_text())
    config["model::compile"] = False
    model = AE(config)
    checkpoints = [entry for entry in trial_dir.iterdir() if entry.name.startswith("checkpoint_")]
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint_* dirs under {trial_dir}")
    latest = max(checkpoints, key=lambda entry: int(entry.name.split("_")[-1]))
    state = torch.load(latest / "state.pt", map_location="cpu")["model"]
    model.load_state_dict({key.replace("_orig_mod.", ""): value for key, value in state.items()})
    model.eval()
    return model, config


def _load_head(head_path: Path) -> nn.Module:
    state = torch.load(head_path, map_location="cpu", weights_only=True)
    if "B" in state:
        B = state["B"].float()
        head = nn.Linear(B.shape[0] - 1, B.shape[1])
        with torch.no_grad():
            head.weight.copy_(B[:-1].T)
            head.bias.copy_(B[-1])
        return head.eval()
    if "0.weight" in state:
        head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )
        with torch.no_grad():
            head[0].weight.copy_(state["0.weight"]); head[0].bias.copy_(state["0.bias"])
            head[2].weight.copy_(state["2.weight"]); head[2].bias.copy_(state["2.bias"])
            head[4].weight.copy_(state["4.weight"]); head[4].bias.copy_(state["4.bias"])
        return head.eval()
    raise KeyError(f"unknown head format in {head_path}: keys {list(state.keys())}")


def _load_canonical_frame(root: Path, manifest: dict):
    """The frame `map_to_canonical=True` aligned to, or `(None, None)` when models stay un-aligned.

    The permutation spec is named rather than unpickled, so the lens loads without SANE's training
    dependencies.
    """
    name = manifest.get("permutation_spec")
    if not name:
        return None, None
    from SANE.git_re_basin import git_re_basin

    checkpoint = torch.load(root / manifest["reference_checkpoint"], map_location="cpu", weights_only=False)
    return checkpoint, getattr(git_re_basin, name)()

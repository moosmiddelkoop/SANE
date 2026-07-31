"""SANEModelWrapper: differentiable weight → SANE embedding → prediction.

Encapsulates the full pipeline for the weight-space unlearning sibling repo
(``WeightSpaceClassifier``):

    flat weight array (4970,)
        →  PyTorch state dict           [diff_weight_pipe.flat_to_state_dict]
        →  token sequence (58, 145)     [diff_weight_pipe.tokenize]
        →  SANE embedding (128,)        [SANE AE.forward_embeddings]
        →  per-class recall (10,)       [prediction_head]

Everything is torch-native and backpropable — no numpy barriers, no
``.data.copy_``, no ``deepcopy`` severing the grad graph.  Gradients flow
from the prediction loss all the way back to the input weight vector, which
is what makes weight-space unlearning possible.

Architecture constants (SmallCNN / MNIST zoo):
    TOTAL_PARAMS  = 4970   flat weight vector length
    n_tokens      = 58     tokens per model
    tokensize     = 145    values per token (= max per-channel size, = ae:i_dim)
    lat_dim       = 128    SANE embedding dimension (= ae:lat_dim)
    n_classes     = 10     per-class recall outputs

Two ways to use this module:

  **Quick path — one-shot export for the unlearning repo:**
    >>> from model_export.model_wapper import export_for_unlearning
    >>> export_for_unlearning(
    ...     ae_checkpoint_path="experiments/smallcnnzoo-mnist/AE_trainable_e550b",
    ...     prediction_head_path="recall_prediction/epoch0-4-8/multivariate_mnist_smallcnnzoo_per_class_recall_head.pt",
    ...     output_pkl_path="meta_network.pkl",
    ... )
    Then in the unlearning repo:
    >>> meta_network = pickle.load(open('meta_network.pkl', 'rb'))
    >>> acc_pred = meta_network(weights.unsqueeze(0)).squeeze(0)  # (10,)

  **Manual path — build the wrapper from components:**
    >>> wrapper = SANEModelWrapper(
    ...     embedding_model=ae_module,        # pretrained AEModule
    ...     prediction_head=pred_head,        # nn.Module: lat_dim → n_classes
    ...     tokensize=config["ae:i_dim"],     # 145
    ...     freeze_encoder=True,              # only input weights get grad
    ... ).to(device).eval()
    >>> recall_pred = wrapper(weights)        # (batch, 10)
    >>> loss = recall_pred.sum()
    >>> loss.backward()                       # grad reaches `weights`
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import torch
import torch.nn as nn

from model_export.diff_weight_pipe import (
    REF_SHAPES,
    TOTAL_PARAMS,
    flat_to_state_dict,
    standardize_state_dict,
    tokenize,
    detokenize,
    state_dict_to_flat,
)


class SANEModelWrapper(nn.Module):
    """Differentiable weight-space model wrapper.

    Input:  flat weight vectors in TF storage order (shape ``(batch, 4970)``
            or ``(4970,)`` for a single model).
    Output: per-class predictions (shape ``(batch, 10)`` or ``(10,)``).

    The forward path is fully differentiable:
        flat → flat_to_state_dict → tokenize → SANE encoder → prediction head

    Key methods:
        forward(weights)       — full pipeline, returns predictions
        encode(weights)        — SANE embedding only (skips prediction head)
        reconstruct(tokens)    — decode tokens back to flat weights (inverse)

    The wrapper only stores the inner ``AE`` model (not the full ``AEModule``
    with its optimizer/criterion/scaler) so it is picklable for the unlearning
    sibling repo.
    """

    def __init__(
        self,
        embedding_model: nn.Module,
        prediction_head: nn.Module,
        *,
        tokensize: int | None = None,
        ignore_bn: bool = False,
        freeze_encoder: bool = False,
        std_stats: dict[str, dict[str, float]] | None = None,
        reference_checkpoint: dict[str, torch.Tensor] | None = None,
        perm_spec=None,
    ):
        """Args:
            embedding_model: Pretrained AEModule (or just the inner AE).
                Must have a ``forward_embeddings(tokens, pos)`` method and a
                ``.config`` dict attribute with at least ``ae:i_dim`` and
                ``ae:lat_dim``.
            prediction_head: nn.Module mapping ``(batch, lat_dim)`` →
                ``(batch, n_classes)``.
            tokensize: Token dimension.  Defaults to ``config["ae:i_dim"]``.
            ignore_bn: Forwarded to tokenize/detokenize (SmallCNN has no BN).
            freeze_encoder: If True, call ``requires_grad_(False)`` on all
                encoder params.  **Permanently modifies the passed model.**
                For unlearning workflows, set this to True so only the
                input weight vector receives gradients.
            std_stats: Per-layer standardization stats, as extracted from
                a cached ``DatasetTokens.layers``.  Dict mapping ``*.weight``
                keys to ``{"mean": float, "std": float}``.  If None, no
                standardization is applied (will hurt performance if the AE
                was trained on standardized tokens).
            reference_checkpoint: Reference model state dict for canonical
                form alignment (from ``DatasetTokens.reference_checkpoint``).
                If set together with ``perm_spec``, each input model is
                aligned to this reference via ``weight_matching`` before
                tokenization.  Required if the AE was trained on
                canonicalized weights.
            perm_spec: ``PermutationSpec`` from
                ``SANE.git_re_basin.git_re_basin`` (e.g.
                ``smallcnnzoo_permutation_spec()``).  Must match the spec
                used during pretraining/preprocessing.
        """
        super().__init__()

        # Keep only the inner AE model — the full AEModule carries an optimizer,
        # criterion, GradScaler, and a consumed parameter generator that are not
        # picklable.  The wrapper only needs forward_embeddings().
        ae_model = getattr(embedding_model, "model", embedding_model)
        self.ae = ae_model
        self.prediction_head = prediction_head

        # Preserve config for lat_dim / i_dim lookups (read-only)
        if hasattr(embedding_model, "config"):
            self.config = embedding_model.config
        elif hasattr(ae_model, "config"):
            self.config = ae_model.config
        else:
            self.config = {}

        # tokensize MUST match the AE's training token dimension (ae:i_dim).
        # If the AE was trained with tokensize=145, we must tokenize at 145
        # so each token lands in the right slice of the tokenizer's input space.
        if tokensize is None:
            try:
                tokensize = self.config["ae:i_dim"]
            except (KeyError, AttributeError):
                tokensize = _discover_tokensize_from_ref()
                warnings.warn(
                    f"tokensize not specified and not found in config; "
                    f"auto-discovered {tokensize}.  Make sure this matches the AE.",
                    stacklevel=2,
                )
        self.tokensize = int(tokensize)
        self.ignore_bn = ignore_bn

        # Pre-compute the position tensor once — it's deterministic for the
        # SmallCNN architecture (58 tokens, 4 layers).  Registered as a
        # persistent buffer so it moves with .to(device) and survives save/load.
        self.register_buffer(
            "_pos",
            _build_fixed_pos(self.tokensize, self.ignore_bn),
            persistent=True,
        )

        # freeze_encoder=True is the default for unlearning: we only want
        # gradients on the input weight vector, not on the SANE encoder.
        # WARNING: this permanently sets requires_grad=False on the passed
        # model's params.  Don't reuse the same AEModule for training.
        if freeze_encoder:
            for p in self.ae.parameters():
                p.requires_grad_(False)

        # Per-layer standardization stats (mean/std per weight key).
        # Applied after flat_to_state_dict and before tokenize, matching
        # DatasetTokens.standardize_data_checkpoints.  None = no standardization.
        self.std_stats = std_stats or {}

        # Canonical form alignment: if both reference_checkpoint and perm_spec
        # are provided, each input model is aligned to the reference via
        # weight_matching (discrete, no grad) then apply_permutation
        # (index_select, grad-safe) before standardization/tokenization.
        # This must match the preprocessing pipeline used during AE pretraining.
        self.reference_checkpoint = reference_checkpoint
        self.perm_spec = perm_spec

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Forward pass — fully differentiable.

        Args:
            weights: ``(batch, TOTAL_PARAMS)`` or ``(TOTAL_PARAMS,)`` float tensor
                     in TF storage order (requires_grad=True for unlearning).

        Returns:
            ``(batch, n_classes)`` predictions (or ``(n_classes,)`` for a single
            sample).

        Note:
            The unlearning repo calls this as
            ``meta_network(weights.unsqueeze(0)).squeeze(0)`` — passing a 2D
            tensor and squeezing the batch dim off the output.
        """
        single = weights.dim() == 1
        tokens = self._weights_to_tokens(weights)          # (B, N, tokensize)
        if tokens.shape[0] == 0:
            return torch.empty(0, self._n_classes(weights), device=weights.device)
        emb = self._tokens_to_embeddings(tokens)           # (B, lat_dim)
        out = self.prediction_head(emb)                    # (B, n_classes)
        return out.squeeze(0) if single else out

    def encode(self, weights: torch.Tensor) -> torch.Tensor:
        """Get SANE embedding without running prediction head.

        Args:
            weights: ``(batch, TOTAL_PARAMS)`` float tensor.

        Returns:
            ``(batch, lat_dim)`` embeddings (or ``(lat_dim,)`` for single sample).
        """
        single = weights.dim() == 1
        tokens = self._weights_to_tokens(weights)
        if tokens.shape[0] == 0:
            return torch.empty(0, self.lat_dim, device=weights.device)
        emb = self._tokens_to_embeddings(tokens)
        return emb.squeeze(0) if single else emb

    def reconstruct(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to flat weight vectors.  Grad-safe.

        Args:
            tokens: ``(batch, N, tokensize)``.

        Returns:
            ``(batch, TOTAL_PARAMS)`` flat tensors.
        """
        pos = self._pos.expand(tokens.shape[0], -1, -1)
        flats = []
        for b in range(tokens.shape[0]):
            sd = detokenize(tokens[b], pos[b], self.ignore_bn)
            flats.append(state_dict_to_flat(sd))
        return torch.stack(flats, dim=0)

    @property
    def n_tokens(self) -> int:
        """Number of tokens per model (fixed for SmallCNN architecture)."""
        return self._pos.shape[0]

    @property
    def lat_dim(self) -> int:
        """Latent dimension from the AE config."""
        return self.config.get("ae:lat_dim", None)

    def _n_classes(self, weights: torch.Tensor | None = None) -> int:
        """Infer n_classes from the prediction head's last Linear layer.

        Used by ``forward`` to size the empty-batch fallback.  The ``weights``
        arg is unused (kept for backward compat).
        """
        try:
            last = [m for m in self.prediction_head.modules()
                    if isinstance(m, nn.Linear)][-1]
            return last.out_features
        except (IndexError, AttributeError):
            return 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _weights_to_tokens(self, weights: torch.Tensor) -> torch.Tensor:
        """Flat weights → stacked token tensor.  Grad-safe.

        Pipeline per model:
            flat → state_dict → [canonicalize] → [standardize] → tokenize
        """
        if weights.dim() == 1:
            weights = weights.unsqueeze(0)

        batch_tokens = []
        for b in range(weights.shape[0]):
            sd = flat_to_state_dict(weights[b])

            # Canonical form alignment (weight_matching + apply_permutation).
            # weight_matching runs on detached weights (discrete optimization,
            # no grad needed).  apply_permutation uses index_select on the
            # grad-enabled tensors, preserving the autograd graph.
            if self.reference_checkpoint is not None and self.perm_spec is not None:
                from SANE.git_re_basin.git_re_basin import (
                    weight_matching, apply_permutation,
                )
                sd_detached = {k: v.detach() for k, v in sd.items()}
                perm = weight_matching(
                    self.perm_spec, self.reference_checkpoint, sd_detached,
                )
                sd = apply_permutation(self.perm_spec, perm, sd)

            if self.std_stats:
                sd = standardize_state_dict(sd, self.std_stats)

            tokens, _mask, _pos = tokenize(
                sd, tokensize=self.tokensize, ignore_bn=self.ignore_bn
            )
            batch_tokens.append(tokens)

        if not batch_tokens:
            return torch.empty(0, self.n_tokens, self.tokensize)
        return torch.stack(batch_tokens, dim=0)

    def _tokens_to_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        """Token tensor → SANE embeddings.  Grad-safe."""
        B = tokens.shape[0]
        pos = self._pos.expand(B, -1, -1).to(device=tokens.device)
        return self.ae.forward_embeddings(tokens, pos.to(torch.int))


# ------------------------------------------------------------------
# Builder helpers
# ------------------------------------------------------------------
def _discover_tokensize_from_ref() -> int:
    """Discover tokensize from REF_SHAPES (no state dict needed)."""
    tsize = 0
    for wkey in REF_SHAPES:
        if "weight" not in wkey:
            continue
        per_chan = REF_SHAPES[wkey].numel() // REF_SHAPES[wkey][0]
        if wkey.replace("weight", "bias") in REF_SHAPES:
            per_chan += 1
        tsize = max(tsize, per_chan)
    return tsize


def _build_fixed_pos(tokensize: int, ignore_bn: bool) -> torch.Tensor:
    """Build a fixed position tensor for the SmallCNN architecture.

    Mirrors the positions that tokenize() produces, keyed off REF_SHAPES.
    """
    # Create a dummy state dict with correct shapes (values don't matter)
    dummy_sd = {}
    for k, shape in REF_SHAPES.items():
        dummy_sd[k] = torch.empty(shape)

    _tokens, _mask, pos = tokenize(dummy_sd, tokensize=tokensize, ignore_bn=ignore_bn)
    return pos


# ------------------------------------------------------------------
# Save / load
# ------------------------------------------------------------------
def save_model_wrapper(wrapper: SANEModelWrapper, path: str | Path) -> None:
    """Save the wrapper to disk as ``model.pt`` + ``metadata.json``.

    Saves:
        - The SANE encoder state dict (from ``wrapper.ae``)
        - The prediction head state dict
        - Metadata (tokensize, lat_dim, n_classes) for reconstruction

    Use ``load_model_wrapper`` to reconstruct.  For the unlearning repo,
    prefer ``export_for_unlearning`` which produces a single pickle file.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save model weights
    torch.save(
        {
            "encoder": getattr(wrapper.ae, "_orig_mod", wrapper.ae).state_dict(),
            "prediction_head": wrapper.prediction_head.state_dict(),
        },
        path / "model.pt",
    )

    # Save metadata
    meta = {
        "tokensize": wrapper.tokensize,
        "ignore_bn": wrapper.ignore_bn,
        "total_params": TOTAL_PARAMS,
    }
    # Try to pull lat_dim and n_classes from the actual heads
    try:
        meta["lat_dim"] = wrapper.config["ae:lat_dim"]
    except Exception:
        meta["lat_dim"] = wrapper.lat_dim
    try:
        last = list(wrapper.prediction_head.modules())[-1]
        if isinstance(last, nn.Linear):
            meta["n_classes"] = last.out_features
    except Exception:
        meta["n_classes"] = 10

    with open(path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def load_model_wrapper(
    path: str | Path,
    config: dict,
    prediction_head: nn.Module,
    device: str = "cpu",
) -> SANEModelWrapper:
    """Load a previously saved wrapper.

    Args:
        path: Directory containing model.pt and metadata.json.
        config: Full SANE config dict (same as used during pretraining).
        prediction_head: nn.Module mapping ``(batch, lat_dim)`` →
            ``(batch, n_classes)``.  Must match the architecture saved.
        device: Target device.

    Returns:
        Reconstructed SANEModelWrapper.
    """
    path = Path(path)
    with open(path / "metadata.json") as f:
        meta = json.load(f)

    # Rebuild AEModule
    from SANE.models.def_AE_module import AEModule

    config = dict(config)  # shallow copy
    config["device"] = device
    config.setdefault("training::steps_per_epoch", 1)
    ae_module = AEModule(config)

    # Load weights
    checkpoint = torch.load(path / "model.pt", map_location=device)
    ae_module.model.load_state_dict(checkpoint["encoder"])

    # Load prediction head weights into the provided module
    prediction_head.load_state_dict(checkpoint["prediction_head"])
    prediction_head = prediction_head.to(device)

    wrapper = SANEModelWrapper(
        embedding_model=ae_module,
        prediction_head=prediction_head,
        tokensize=meta["tokensize"],
        ignore_bn=meta.get("ignore_bn", False),
    ).to(device)

    return wrapper


# ------------------------------------------------------------------
# Export for unlearning sibling repo
#
# This is the recommended entry point for collaborators.  It loads the
# pretrained AE + prediction head from their on-disk locations, builds the
# wrapper, and pickles it into a single file that the unlearning repo can
# ``pickle.load``.  The resulting pickle is ~0.8 GB (dominated by the AE
# weights: 202M parameters).
#
# The alternative is save_model_wrapper/load_model_wrapper above, which use
# a directory with separate model.pt + metadata.json.  Use those if you need
# to swap prediction heads without reloading the AE.
# ------------------------------------------------------------------
def export_for_unlearning(
    ae_checkpoint_path: str | Path,
    prediction_head_path: str | Path,
    output_pkl_path: str | Path = "meta_network.pkl",
    *,
    dataset_cache_path: str | Path | None = None,
    device: str = "cpu",
    freeze_encoder: bool = True,
    canonicalize: bool = False,
) -> Path:
    """Build a SANEModelWrapper and pickle it for the unlearning sibling repo.

    One-shot convenience that:
        1) Loads the pretrained AE from ``ae_checkpoint_path``
        2) Loads the prediction head from ``prediction_head_path``
        3) Loads per-layer standardization stats from ``dataset_cache_path``
        4) Wraps them in a ``SANEModelWrapper`` with ``freeze_encoder=True``
        5) Pickles the wrapper to ``output_pkl_path``

    The resulting ``.pkl`` file can be loaded by the unlearning pipeline:
        >>> import pickle
        >>> meta_network = pickle.load(open('meta_network.pkl', 'rb'))
        >>> acc_pred = meta_network(weights.unsqueeze(0)).squeeze(0)

    Args:
        ae_checkpoint_path: Path to a SANE pretraining checkpoint dir
            (the one containing ``params.json`` and ``checkpoint_XXXXXX/state.pt``).
            Example: ``"experiments/smallcnnzoo-mnist/AE_trainable_e550b"``.
        prediction_head_path: Path to the trained prediction head ``.pt`` file.
            Example: ``"recall_prediction/epoch0-4-8/multivariate_mnist_smallcnnzoo_per_class_recall_head.pt"``.
        output_pkl_path: Where to write the pickled wrapper.
        dataset_cache_path: Path to a cached ``DatasetTokens`` ``.pt`` file
            (e.g. ``"recall_prediction/epoch0-4-8/dataset_cache/smallcnnzoo_mnist_epoch0-4-8_train.pt"``).
            The per-layer mean/std stats are extracted from ``ds.layers`` and
            embedded into the wrapper.  If None, no standardization is applied
            (will hurt performance if the AE was trained on standardized tokens).
        device: ``"cpu"`` or ``"cuda"``.
        freeze_encoder: Passed through to ``SANEModelWrapper``.
        canonicalize: If True, extract the reference checkpoint and
            permutation spec from the cached dataset and enable
            ``weight_matching`` alignment per forward pass.  Adds compute
            overhead; empirically does not improve R² for the SmallCNN/MNIST
            zoo (the models are already roughly aligned).  Default False.

    Returns:
        Absolute path to the written ``.pkl`` file.
    """
    import pickle
    from SANE.models.def_AE_module import AEModule

    ae_checkpoint_path = Path(ae_checkpoint_path)

    # ---- 1. Load AE config and checkpoint ----
    config = json.load((ae_checkpoint_path / "params.json").open("r"))
    config["device"] = device
    config["training::steps_per_epoch"] = 1
    config["model::compile"] = False

    print(f"Loading AE from {ae_checkpoint_path} ...")
    ae_module = AEModule(config)

    # Find the highest-numbered checkpoint directory
    ckpt_dirs = sorted(
        [d for d in ae_checkpoint_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint_")],
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if not ckpt_dirs:
        raise FileNotFoundError(f"No checkpoint_* dirs found under {ae_checkpoint_path}")
    ckpt_dir = ckpt_dirs[-1]
    print(f"  using checkpoint: {ckpt_dir.name}")

    checkpoint = torch.load(ckpt_dir / "state.pt", map_location=device)
    model_state = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    ae_module.model.load_state_dict(model_state, strict=True)
    ae_module.model.eval()

    # ---- 2. Load prediction head ----
    print(f"Loading prediction head from {prediction_head_path} ...")
    head_data = torch.load(prediction_head_path, map_location=device)
    B = head_data["B"]                         # (lat_dim + 1, n_classes)
    lat_dim = config["ae:lat_dim"]             # 128
    n_classes = B.shape[1]                     # 10

    pred_head = nn.Linear(lat_dim, n_classes, bias=True)
    pred_head.weight.data = B[:lat_dim].T.float()
    pred_head.bias.data = B[lat_dim].float()
    pred_head.eval()

    # ---- 3. Load standardization stats + (optionally) reference model from cached DatasetTokens ----
    std_stats = None
    reference_checkpoint = None
    perm_spec = None
    if dataset_cache_path is not None:
        dataset_cache_path = Path(dataset_cache_path)
        print(f"Loading preprocessing metadata from {dataset_cache_path} ...")
        ds = torch.load(dataset_cache_path, map_location="cpu", weights_only=False)
        std_stats = ds.layers
        for key, st in std_stats.items():
            print(f"  std {key}: mean={st['mean']:.6f}, std={st['std']:.6f}")

        if canonicalize:
            if hasattr(ds, "reference_checkpoint") and ds.reference_checkpoint is not None:
                reference_checkpoint = ds.reference_checkpoint
                print(f"  reference_checkpoint: {len(reference_checkpoint)} keys")
            if ds.permutation_spec is not None:
                perm_spec = ds.permutation_spec
                print(f"  perm_spec: {type(perm_spec).__name__}")

    # ---- 4. Build wrapper ----
    print(f"Building wrapper (tokensize={config['ae:i_dim']}, "
          f"freeze_encoder={freeze_encoder}, "
          f"standardize={'yes' if std_stats else 'no'}, "
          f"canonicalize={'yes' if reference_checkpoint else 'no'}) ...")
    wrapper = SANEModelWrapper(
        embedding_model=ae_module,
        prediction_head=pred_head,
        tokensize=config["ae:i_dim"],
        freeze_encoder=freeze_encoder,
        std_stats=std_stats,
        reference_checkpoint=reference_checkpoint,
        perm_spec=perm_spec,
    ).to(device).eval()

    # ---- 5. Pickle ----
    output_pkl_path = Path(output_pkl_path).absolute()
    print(f"Pickling wrapper to {output_pkl_path} ...")
    with open(output_pkl_path, "wb") as f:
        pickle.dump(wrapper, f)

    print(f"Done.  File size: {output_pkl_path.stat().st_size / 1e9:.2f} GB")
    return output_pkl_path

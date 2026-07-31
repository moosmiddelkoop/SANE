"""Check that the sane_wse lens is the same function as the exported meta_network.pkl.

The recorded SANE unlearning results were produced with `model_export.model_wapper.SANEModelWrapper`,
pickled by `export_for_unlearning`. The lens replaces its per-model Python loop with one static
gather, so predictions must agree, and the batched path must be faster.

Run from the repo root:
    .venv/bin/python3 sane_wse/test_vs_export.py <zoo_dir> [n_models]
"""

import pickle
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import torch

_repo = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo))

# model_export.reconstruct_network imports pytorch_port, which imports keras. Unpickling the
# wrapper needs the module to import, not to work, so the import is stubbed out.
sys.modules["pytorch_port"] = MagicMock()
sys.modules["pytorch_port.model"] = MagicMock()

from model_export.diff_weight_pipe import state_dict_to_flat  # noqa: E402
from sane_wse import from_manifest  # noqa: E402

MANIFEST = "sane_wse_manifest.yaml"
EXPORTED = "model_export/meta_network.pkl"
EPOCH = "checkpoint_000008"  # the final checkpoint, which is what the recorded runs used


def load_models(zoo, n_models):
    """The first `n_models` final checkpoints, in wse's native state_dict order (weight, bias)."""
    directories = sorted(entry for entry in Path(zoo).iterdir() if (entry / "params.json").is_file())[:n_models]
    models = []
    for directory in directories:
        checkpoint = torch.load(directory / EPOCH / "checkpoints", map_location="cpu", weights_only=False)
        models.append({f"{layer}.{part}": checkpoint[f"{layer}.{part}"] for layer in ["conv1", "conv2", "conv3", "dense"] for part in ("weight", "bias")})
    return models


def main():
    zoo = sys.argv[1]
    n_models = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    models = load_models(zoo, n_models)
    params = {name: torch.stack([model[name] for model in models]) for name in models[0]}
    shapes = {name: tensor.shape for name, tensor in models[0].items()}

    lens = from_manifest(MANIFEST)
    lens.prepare(shapes, torch.device("cpu"))
    lens.eval().requires_grad_(False)

    with open(EXPORTED, "rb") as handle:
        exported = pickle.load(handle)
    exported.eval()
    print(f"exported wrapper: tokensize={exported.tokensize} n_tokens={exported.n_tokens} "
          f"canonicalize={exported.reference_checkpoint is not None and exported.perm_spec is not None}")
    assert exported.std_stats.keys() == lens.standardization.keys(), "standardization stats differ from the export"
    for key, stat in exported.std_stats.items():
        assert stat == lens.standardization[key], f"{key}: {stat} != {lens.standardization[key]}"
    print("OK  standardization stats are the ones the export used")

    flat = torch.stack([state_dict_to_flat(model) for model in models])
    with torch.no_grad():
        start = time.time()
        reference = exported(flat)
        exported_seconds = time.time() - start
        start = time.time()
        prediction = lens(params)
        lens_seconds = time.time() - start

    gap = float((prediction - reference).abs().max())
    print(f"OK  predictions agree to {gap:.3e} over {n_models} models x {lens.n_classes} classes")
    assert gap < 1e-4, "the lens does not reproduce the exported meta-network"

    print(f"    exported {exported_seconds / n_models:.3f} s/model, lens {lens_seconds / n_models:.3f} s/model "
          f"({exported_seconds / lens_seconds:.1f}x)")
    negative = int((reference < 0).any(dim=1).sum())
    print(f"    {negative}/{n_models} models have at least one negative predicted class accuracy")


if __name__ == "__main__":
    main()

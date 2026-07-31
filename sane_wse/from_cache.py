"""Extract what the wse side needs out of the cached DatasetTokens objects.

The caches are the only record of how the AE and the recall head saw the zoo: which models landed
in which split, the per-layer standardization statistics, and the reference model that
`map_to_canonical=True` aligned to. They are large pickles that need SANE's full training
dependencies, so this reads them once and writes three small files instead:

    splits.json             the split file `wse run --split-file` reads
    standardization.yaml    the block to paste into the lens manifest
    reference_checkpoint.pt the canonical frame, for a manifest that names a permutation spec

Run from a repo root that has the caches:
    .venv/bin/python3 sane_wse_from_cache.py recall_prediction/epoch0-4-8/dataset_cache <out_dir>
"""

import json
import sys
from pathlib import Path

import torch
import yaml

SPLITS = ["train", "val", "test"]


def load(cache_dir, split):
    [cache] = sorted(Path(cache_dir).glob(f"*_{split}.pt"))
    return torch.load(cache, map_location="cpu", weights_only=False)


def main():
    cache_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)

    splits = {}
    for split in SPLITS:
        dataset = load(cache_dir, split)
        splits[split] = sorted(Path(path).name for path in dataset.path_list)
        print(f"{split}: {len(splits[split])} models")
        if split == "train":  # the split the head was fit on, so its stats are the ones in use
            standardization = {key: dict(stat) for key, stat in dataset.layers.items()}
            reference_checkpoint = dataset.reference_checkpoint

    (out / "splits.json").write_text(json.dumps(splits, indent=1))
    (out / "standardization.yaml").write_text(yaml.safe_dump({"standardization": standardization}, sort_keys=False))
    torch.save(reference_checkpoint, out / "reference_checkpoint.pt")
    print(f"wrote splits.json, standardization.yaml and reference_checkpoint.pt to {out}")


if __name__ == "__main__":
    main()

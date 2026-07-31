"""Recover which zoo models were in train/val/test of the recall-MLP runs.

The embeddings caches store per-model targets Y (per-class recall at zoo
epoch 8) and model ids, but not the model paths. Line 8 of each zoo model's
result.json (training_iteration 8, idx_offset=0) holds the same ten
acc_class_i values, so a Y row identifies its model directory. The
train/val/test slicing is reproduced exactly with random.Random(67), as in
the property_prediction_*_smallcnnzoo_mlp.py scripts.

Writes recall_prediction/mlp/split.json per zoo. Entries are model directory
names; a list means the fingerprint is ambiguous (candidates), null means no
match was found.

Run via recover_recall_mlp_split.sh (sbatch).
"""

import json
import logging
import random
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)

SEED = 67
EPOCH_SET = "8"
EPOCH_LINE = 8  # result.json line with training_iteration 8 (idx_offset=0)
EXP = Path("/gpfs/home1/mmiddelkoop/SANE/experiments")
ZOO_ROOT = Path("/gpfs/scratch1/shared/mmiddelkoop/unthi_zoo")

# cifar10 uses the corrected 3-way split of the rerun (the original run used
# [0.7, 0.3], which left the test set empty).
ZOOS = {
    "svhn": {
        "cache": EXP / "smallcnnzoo-svhn/recall_prediction/mlp/embeddings.pt",
        "zoo": ZOO_ROOT / "unthi_svhn",
        "ds_split": [0.7, 0.15, 0.15],
        "out": EXP / "smallcnnzoo-svhn/recall_prediction/mlp/split.json",
    },
    "cifar10": {
        "cache": EXP / "smallcnnzoo-cifar10/recall_prediction/mlp/embeddings.pt",
        "zoo": ZOO_ROOT / "unthi_cifar10",
        "ds_split": [0.7, 0.15, 0.15],
        "out": EXP / "smallcnnzoo-cifar10/recall_prediction/mlp/split.json",
    },
    "fmnist": {
        "cache": EXP / "smallcnnzoo-fmnist/recall_prediction/mlp/embeddings.pt",
        "zoo": ZOO_ROOT / "unthi_fmnist",
        "ds_split": [0.7, 0.15, 0.15],
        "out": EXP / "smallcnnzoo-fmnist/recall_prediction/mlp/split.json",
    },
    "mnist": {
        "cache": EXP / "smallcnnzoo-mnist/recall_prediction/mse_spread/embeddings.pt",
        "zoo": ZOO_ROOT / "unthi_mnist",
        "ds_split": [0.7, 0.15, 0.15],
        "out": EXP / "smallcnnzoo-mnist/recall_prediction/mlp/split.json",
    },
}


def fingerprint(values):
    """Ten recall values -> hashable key. NaN / -999 / missing -> 'nan'.

    Both sides pass through float32, so exact equality holds: the cache Y was
    built with torch.tensor(json_value, dtype=torch.float).
    """
    key = []
    for v in values:
        if v is None or v == -999.0 or v != v:
            key.append("nan")
        else:
            key.append(float(np.float32(v)))
    return tuple(key)


def zoo_fingerprints(zoo_dir):
    fp2names = {}
    dirs = [d for d in zoo_dir.iterdir() if d.is_dir()]
    for n, d in enumerate(dirs):
        if n % 5000 == 0:
            logging.info(f"{zoo_dir.name}: {n}/{len(dirs)} result.json read")
        try:
            line = json.loads((d / "result.json").read_text().splitlines()[EPOCH_LINE])
        except Exception as e:
            logging.warning(f"skip {d.name}: {e}")
            continue
        fp = fingerprint([line.get(f"acc_class_{i}") for i in range(10)])
        fp2names.setdefault(fp, []).append(d.name)
    return fp2names


for zoo, cfg in ZOOS.items():
    logging.info(f"=== {zoo}")
    cache = torch.load(cfg["cache"], map_location="cpu")[EPOCH_SET]
    Y, mid = cache["Y"], cache["mid"]
    n_models = int(mid.max()) + 1
    first_row = {}
    for i, m in enumerate(mid.tolist()):
        first_row.setdefault(m, i)

    fp2names = zoo_fingerprints(cfg["zoo"])

    names, n_unmatched, n_ambiguous = [], 0, 0
    for m in range(n_models):
        candidates = fp2names.get(fingerprint(Y[first_row[m]].tolist()))
        if candidates is None:
            names.append(None)
            n_unmatched += 1
        elif len(candidates) == 1:
            names.append(candidates[0])
        else:
            names.append(candidates)
            n_ambiguous += 1

    models = list(range(n_models))
    random.Random(SEED).shuffle(models)
    idx1 = int(cfg["ds_split"][0] * n_models)
    idx2 = idx1 + int(cfg["ds_split"][1] * n_models)
    split = {
        "seed": SEED,
        "ds_split": cfg["ds_split"],
        "epoch_set": EPOCH_SET,
        "n_models": n_models,
        "n_unmatched": n_unmatched,
        "n_ambiguous": n_ambiguous,
        "train": [names[m] for m in models[:idx1]],
        "val": [names[m] for m in models[idx1:idx2]],
        "test": [names[m] for m in models[idx2:]],
    }
    with cfg["out"].open("w") as f:
        json.dump(split, f, indent=4)
    logging.info(
        f"{zoo}: {n_models} models, {n_unmatched} unmatched, "
        f"{n_ambiguous} ambiguous -> {cfg['out']}"
    )

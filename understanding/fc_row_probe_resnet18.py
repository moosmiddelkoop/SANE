"""fc-row probe: is the model x class part of per-class recall readable from the fc layer?

ResNet-18 CIFAR-100 zoo, epochs 5..60 (the samples of the recall-transformer token cache).
Features per (model, class c): fc.bias[c] and the norm of fc.weight[c]. Both do not change under
the weight permutations, so the raw checkpoints are used (no canonicalization, no standardization).
Per epoch, the targets and the features are double-centered within each split (remove the model
means and the class means), as r2_interaction in train_recall_transformer_resnet18.py does.
One linear regression, shared by all classes, is fit on train and scored on val and test.
The recall transformers get r2_interaction ~ 0 here.
Run: sbatch fc_row_probe_resnet18.sh
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform")
CACHE_META = Path("/scratch-shared/mmiddelkoop/recall_transformer/resnet18_cifar100/tokens_ep5-60_meta.pt")
FEATURES_PATH = CACHE_META.parent / "fc_row_features.pt"
OUTPUT_PATH = Path(__file__).parent / "fc_row_probe_resnet18.json"
FEATURE_SETS = {"bias": [0], "norm": [1], "bias+norm": [0, 1]}

meta = torch.load(CACHE_META, weights_only=False)
Y, epochs, splits = meta["Y"].double(), torch.as_tensor(meta["epoch"]), list(meta["split"])


def fc_row_features(i):
    checkpoint_path = ZOO_ROOT / meta["model_dir"][i] / f"checkpoint_{int(epochs[i]):06d}" / "checkpoints"
    state = torch.load(checkpoint_path, map_location="cpu", mmap=True)  # mmap: reads only the fc bytes
    return torch.stack([state["fc.bias"], state["fc.weight"].norm(dim=1)], dim=1)  # [100 classes, 2]


if FEATURES_PATH.exists():
    features = torch.load(FEATURES_PATH)
else:
    with ThreadPoolExecutor(16) as pool:  # GPFS reads are slow per reader, so read in parallel
        features = torch.stack(list(pool.map(fc_row_features, range(len(Y))))).double()  # [N, 100, 2]
    torch.save(features, FEATURES_PATH)


def double_center(x):
    """[models, classes, ...] -> remove the model means and the class means"""
    return x - x.mean(dim=1, keepdim=True) - x.mean(dim=0, keepdim=True) + x.mean(dim=(0, 1), keepdim=True)


results = {}
for epoch in range(5, 61, 5):
    rows = {split: (epochs == epoch) & torch.tensor([s == split for s in splits]) for split in ["train", "val", "test"]}
    targets = {split: double_center(Y[r]) for split, r in rows.items()}
    centered_features = {split: double_center(features[r]) for split, r in rows.items()}
    results[epoch] = {}
    for name, columns in FEATURE_SETS.items():
        x_train = centered_features["train"][..., columns].reshape(-1, len(columns))
        weights = torch.linalg.lstsq(x_train, targets["train"].reshape(-1, 1)).solution
        results[epoch][name] = {}
        for split in ["train", "val", "test"]:
            predictions = (centered_features[split][..., columns] @ weights).squeeze(-1)
            r2 = 1 - ((targets[split] - predictions) ** 2).sum() / (targets[split] ** 2).sum()
            results[epoch][name][f"r2_interaction_{split}"] = round(r2.item(), 4)
    print(epoch, {name: r["r2_interaction_test"] for name, r in results[epoch].items()}, flush=True)

OUTPUT_PATH.write_text(json.dumps(results, indent=2))

"""Per-class recall from the last-layer weights of the full CIFAR-100 ResNet-18 zoo.

Tests whether the between-model spread of per-class recall is more predictable in the
middle of training than at the end. For every 5th epoch (5..60) a ridge head and an
MLP head predict acc_class_0..99 of a checkpoint from the final layer (fc) of the same
checkpoint. R^2 is per class over the models, so it only measures the between-model
differences, then averaged over the 100 classes. The class-centered R^2 pools the
squared errors of all (model, class) pairs and divides by their squared distance to the
mean of their class. So a class with a large spread counts more than a class with a
small spread.

Features: fc.bias (100) and the squared norm of each class row of fc.weight (100). Both do
not change under a permutation of the 512 penultimate units, which differs between seeds.
Features and targets are standardized with the statistics of the fitting data.
A first run used the full Gram matrix fc.weight @ fc.weight.T (5150 features for 700 train
models); every head then predicted no better than the mean (results in
recall_prediction/last_layer/*_gram.*).

Splits from the zoo's split.json (the same as the epoch-60 zoo):
- ridge: fit on train, regularizer chosen on val, scored on test
- own-class linear: per class, least squares on only that class's bias and row norm,
  fit on train, scored on test
- MLP (eval_per_class_recall_MLP): fit on train, the epoch with the lowest val loss
  kept (early stopping), scored on test, one run per seed

Features and targets are cached in recall_prediction/last_layer/features.pt.
Run with the sibling .sh.
"""

import os

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

import json
import logging
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from SANE.datasets.zoo_split import load_split
from SANE.models.def_downstream_module import DownstreamTaskLearner
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform")
OUT_DIR = Path("recall_prediction/last_layer")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_PT = OUT_DIR / "features.pt"
RESULTS_JSON = OUT_DIR / "per_class_recall_last_layer.json"

EPOCHS = list(range(5, 61, 5))
NUM_CLASSES = 100
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(NUM_CLASSES)]
SPLITS = ["train", "val", "test"]
SEED = 42
MLP_SEEDS = [0, 1, 2]
MLP_EPOCHS = 200
LR = 1e-3
MLP_BATCH_SIZE = 64

seed_everything(SEED)
split = load_split(ZOO_ROOT)


def last_layer_features(checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu", mmap=True)
    weight, bias = state["fc.weight"].double(), state["fc.bias"].double()
    return torch.cat([bias, (weight**2).sum(dim=1)])


def recall_targets(model_dir):
    lines = [json.loads(line) for line in (model_dir / "result.json").read_text().splitlines()]
    line_of_epoch = {line["training_iteration"]: line for line in lines}
    return torch.tensor([[line_of_epoch[epoch][key] for key in ACC_CLASS_KEYS] for epoch in EPOCHS])


if FEATURES_PT.exists():
    logging.info(f"Loading cached features from {FEATURES_PT}")
    data = torch.load(FEATURES_PT)
else:
    data = {"split_id": split.split_id}
    for split_name in SPLITS:
        model_dirs = split.paths(split_name)
        checkpoint_paths = [d / f"checkpoint_{epoch:06d}" / "checkpoints" for d in model_dirs for epoch in EPOCHS]
        logging.info(f"{split_name}: loading {len(checkpoint_paths)} checkpoints")
        with ThreadPoolExecutor(16) as pool:
            features = torch.stack(list(pool.map(last_layer_features, checkpoint_paths)))
        data[split_name] = {
            "features": features.view(len(model_dirs), len(EPOCHS), -1),  # (models, epochs, 200)
            "targets": torch.stack([recall_targets(d) for d in model_dirs]),  # (models, epochs, 100)
        }
    torch.save(data, FEATURES_PT)
assert data["split_id"] == split.split_id, f"{FEATURES_PT} is from another split; delete it"


def standardize(fit_tensor, *tensors):
    mean, std = fit_tensor.mean(dim=0), fit_tensor.std(dim=0)
    return [((t - mean) / std).float() for t in (fit_tensor, *tensors)]


def class_centered_r2(predictions, targets):
    """R^2 pooled over all (model, class) pairs, around the mean of each class."""
    targets = targets.double()
    return (1 - ((targets - predictions) ** 2).sum() / ((targets - targets.mean(dim=0)) ** 2).sum()).item()


dstk = DownstreamTaskLearner()
dstk.polar_coordinates = False
dstk.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results = {}
for epoch_index, epoch in enumerate(EPOCHS):
    X = {s: data[s]["features"][:, epoch_index] for s in SPLITS}
    Y = {s: data[s]["targets"][:, epoch_index] for s in SPLITS}

    # ridge: R^2 does not depend on the target scale, so only the features are standardized
    X_train, X_val, X_test = standardize(X["train"], X["val"], X["test"])
    ridge = dstk.compute_closed_form_solution_multivariate(
        z_train=X_train, Y_train=Y["train"], z_test=X_test, Y_test=Y["test"], z_val=X_val, Y_val=Y["val"]
    )
    ridge_predictions = torch.cat([X_test, torch.ones(len(X_test), 1)], dim=1).double() @ ridge["B"].cpu()
    ridge_summary = {
        "reg_best": ridge["reg_best"],
        "r2_train": ridge["r2_train"].mean().item(),
        "r2_val": ridge["r2_val"].mean().item(),
        "r2_test": ridge["r2_test"].mean().item(),
        "class_centered_r2_test": class_centered_r2(ridge_predictions, Y["test"]),
        "r2_test_per_class": ridge["r2_test"].tolist(),
    }

    # linear fit per class on only its own two features (bias and row norm), fit on train
    own_class_predictions = []
    for c in range(NUM_CLASSES):
        own_features = lambda X_split: torch.stack(
            [X_split[:, c], X_split[:, NUM_CLASSES + c], torch.ones(len(X_split))], dim=1
        ).double()
        coefficients = torch.linalg.lstsq(own_features(X_train), Y["train"][:, c : c + 1].double()).solution
        own_class_predictions.append(own_features(X_test) @ coefficients)
    own_class_predictions = torch.cat(own_class_predictions, dim=1)
    own_class_r2_test = dstk.compute_r2_multivariate(own_class_predictions, Y["test"].double())
    own_class_summary = {
        "r2_test": own_class_r2_test.mean().item(),
        "class_centered_r2_test": class_centered_r2(own_class_predictions, Y["test"]),
        "r2_test_per_class": own_class_r2_test.tolist(),
    }

    # MLP: fit on train, early stopping on val; standardized targets, else the MLP only
    # learns the class means
    Y_mean, Y_std = Y["train"].mean(dim=0), Y["train"].std(dim=0)
    Y_train, Y_val, Y_test = [((Y[s] - Y_mean) / Y_std).float() for s in SPLITS]
    per_seed = {}
    for seed in MLP_SEEDS:
        seed_everything(seed)
        result = dstk.eval_per_class_recall_MLP(
            model=None,
            trainset=(X_train, Y_train),
            valset=(X_val, Y_val),
            testset=(X_test, Y_test),
            target_keys=ACC_CLASS_KEYS,
            epochs=MLP_EPOCHS,
            lr=LR,
            mlp_batch_size=MLP_BATCH_SIZE,
        )
        with torch.no_grad():
            mlp_predictions = result["mlp"](X_test.to(dstk.device)).cpu().double() * Y_std + Y_mean
        per_seed[seed] = {
            "r2_train": result["r2_train"],
            "r2_test": result["r2_test"],
            "class_centered_r2_test": class_centered_r2(mlp_predictions, Y["test"]),
            "best_epoch": result["best_epoch"],
        }
    mlp_summary = {"per_seed": per_seed}
    for key in ["r2_train", "r2_test", "class_centered_r2_test"]:
        mlp_summary[f"{key}_mean"] = statistics.mean(r[key] for r in per_seed.values())
        mlp_summary[f"{key}_std"] = statistics.stdev(r[key] for r in per_seed.values())

    results[epoch] = {"ridge": ridge_summary, "own_class_linear": own_class_summary, "mlp": mlp_summary}
    logging.info(
        f"epoch {epoch:2d} R^2 test (mean per class / class-centered): "
        f"ridge {ridge_summary['r2_test']:.3f} / {ridge_summary['class_centered_r2_test']:.3f} "
        f"(reg {ridge_summary['reg_best']}); "
        f"own-class linear {own_class_summary['r2_test']:.3f} / {own_class_summary['class_centered_r2_test']:.3f}; "
        f"MLP {mlp_summary['r2_test_mean']:.3f} / {mlp_summary['class_centered_r2_test_mean']:.3f} "
        f"(train {mlp_summary['r2_train_mean']:.3f}, best epochs {[r['best_epoch'] for r in per_seed.values()]})"
    )

summary = {
    "zoo": str(ZOO_ROOT),
    "split_id": split.split_id,
    "features": "fc.bias + squared row norms of fc.weight",
    "mlp": {"epochs": MLP_EPOCHS, "lr": LR, "mlp_batch_size": MLP_BATCH_SIZE, "seeds": MLP_SEEDS},
    "per_epoch": results,
}
RESULTS_JSON.write_text(json.dumps(summary, indent=4))
logging.info(f"Wrote results to {RESULTS_JSON}")

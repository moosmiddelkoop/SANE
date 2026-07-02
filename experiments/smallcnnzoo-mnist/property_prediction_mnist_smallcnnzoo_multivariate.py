"""Per-class recall (acc_class_0..9) property prediction for the smallcnnzoo-mnist SANE model.

Most-elementary setup:
  * The pretrained SANE autoencoder encodes each model checkpoint into a 128-dim embedding.
  * For each of the 10 per-class recall targets we fit an independent closed-form ridge
    regression from embedding -> recall (DownstreamTaskLearner). 10 independent scalar
    regressions == one linear head with 10 outputs (least squares decouples per output).
  * Per-class recall is only valid at training_iteration 0, 4, 8 (-999 elsewhere), so we
    load ONLY those epochs and there is nothing to filter.

All DatasetTokens preprocessing params are kept identical to
data/preprocess_dataset_smallcnnzoo_mnist.py so the embeddings stay on-distribution.

Run from this directory with the project venv:
    source "$HOME/SANE/.venv/bin/activate"
    python property_prediction_mnist_smallcnnzoo_multivariate.py
"""

import logging
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import json
from pathlib import Path

import torch

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec
from SANE.models.def_AE_module import AEModule
from SANE.models.def_downstream_module import DownstreamTaskLearner
from SANE.models.downstream_baselines import LayerQuintiles

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
TRIAL_DIR = Path(
    "sane_pretraining/sane_mnist_smallcnnzoo/"
    "AE_trainable_e550b_00000_0_2026-06-24_17-46-42"
)
OUT_DIR = Path("epoch0-4-8")
CHECKPOINT = TRIAL_DIR / "checkpoint_000010" / "state.pt"   # latest available checkpoint
ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist/")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_JSON = OUT_DIR / "multivariate_mnist_smallcnnzoo_per_class_recall.json"

EPOCH_LIST = [8]  # the only iterations with valid acc_class_* values
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load pretrained SANE autoencoder
# ---------------------------------------------------------------------------
logging.info("Loading pretrained SANE model")
config = json.load((TRIAL_DIR / "params.json").open("r"))
config["device"] = device
config["model::compile"] = False  # no need to compile for inference

config["training::steps_per_epoch"] = 1
module = AEModule(config)

checkpoint = torch.load(CHECKPOINT, map_location=device)
# the trial was trained with model::compile=True, so keys carry an "_orig_mod." prefix
state_dict = {
    k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
}
module.model.load_state_dict(state_dict)
module.model.eval()

# ---------------------------------------------------------------------------
# Build DatasetTokens over the zoo at epochs [0, 4, 8]
# (params identical to preprocess_dataset_smallcnnzoo_mnist.py)
# ---------------------------------------------------------------------------
property_keys = {
    "result_keys": ACC_CLASS_KEYS + ["test_acc", "training_iteration"],
    "config_keys": [],
}


def build_split(split):
    logging.info(f"Building DatasetTokens split={split}")
    return DatasetTokens(
        root=ZOO_ROOT,
        epoch_lst=EPOCH_LIST,
        mode="vector",
        permutation_spec=smallcnnzoo_permutation_spec(),
        map_to_canonical=True,
        standardize=True,
        tokensize=config["ae:i_dim"],          # 145
        train_val_test=split,
        ds_split=[0.7, 0.15, 0.15],
        weight_threshold=100,
        max_samples=None,
        property_keys=property_keys,
        shuffle_path=True,
        num_threads=12,
        verbosity=3,
        getitem="tokens+props",
        ignore_bn=True,
    )


ds_train = build_split("train")
ds_val = build_split("val")
ds_test = build_split("test")

# ---------------------------------------------------------------------------
# Multivariate linear (closed-form ridge) head: 10 outputs (per-class recall),
# fit as a single solve with one shared regularizer. Returns the fitted head B
# ([D+1, 10]) so it can be frozen and backpropagated through for weight-space
# surgery later (y_hat = [z, 1] @ B).
# ---------------------------------------------------------------------------
logging.info("Fitting multivariate per-class recall head on SANE embeddings")
dstk = DownstreamTaskLearner()
result = dstk.eval_multivariate_regression(
    model=module,
    trainset=ds_train,
    testset=ds_test,
    valset=ds_val,
    target_keys=ACC_CLASS_KEYS,
    batch_size=256,
)

per_class_test = dict(zip(ACC_CLASS_KEYS, [r.item() for r in result["r2_test"]]))
logging.info(f"reg_best: {result['reg_best']}")
logging.info(f"Per-class test R^2: {per_class_test}")
logging.info(f"Mean test R^2: {result['mean_r2_test']:.4f}")

# --- baseline: same multivariate head on LayerQuintiles weight statistics ---
logging.info("Fitting baseline (LayerQuintiles) per-class recall head")
lq_result = dstk.eval_multivariate_regression(
    model=LayerQuintiles(),
    trainset=ds_train,
    testset=ds_test,
    valset=ds_val,
    target_keys=ACC_CLASS_KEYS,
    batch_size=256,
)
logging.info(f"Baseline (LayerQuintiles) mean test R^2: {lq_result['mean_r2_test']:.4f}")


def _r2_dict(res, split):
    return dict(zip(ACC_CLASS_KEYS, [v.item() for v in res[f"r2_{split}"]]))


# --- persist results ---
summary = {
    "method": "sane_mnist_smallcnnzoo",
    "epoch_list": EPOCH_LIST,
    "reg_best": result["reg_best"],
    "sane": {
        "r2_train": _r2_dict(result, "train"),
        "r2_val": _r2_dict(result, "val"),
        "r2_test": _r2_dict(result, "test"),
        "mean_r2_test": result["mean_r2_test"],
    },
    "layerquintiles_baseline": {
        "r2_test": _r2_dict(lq_result, "test"),
        "mean_r2_test": lq_result["mean_r2_test"],
    },
}
with open(RESULTS_JSON, "w") as f:
    json.dump(summary, f, indent=4)
logging.info(f"Wrote results to {RESULTS_JSON}")

# --- persist the frozen linear head B for weight-space surgery ---
HEAD_PATH = OUT_DIR / "multivariate_mnist_smallcnnzoo_per_class_recall_head.pt"
torch.save(
    {
        "B": result["B"].cpu(),  # [D+1, 10], last row is the bias
        "reg_best": result["reg_best"],
        "target_keys": ACC_CLASS_KEYS,
        "epoch_list": EPOCH_LIST,
    },
    HEAD_PATH,
)
logging.info(f"Saved frozen per-class recall head to {HEAD_PATH}")

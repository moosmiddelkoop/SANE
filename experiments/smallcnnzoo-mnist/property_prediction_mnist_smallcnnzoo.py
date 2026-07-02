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
    python property_prediction_mnist_smallcnnzoo.py
"""

import logging
import os

# TODO: understand this
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

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
TRIAL_DIR = Path(
    "sane_pretraining/sane_mnist_smallcnnzoo/"
    "AE_trainable_e550b_00000_0_2026-06-24_17-46-42"
)
CHECKPOINT = TRIAL_DIR / "checkpoint_000010" / "state.pt"   # latest available checkpoint
ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist/")
RESULTS_JSON = "mnist_smallcnnzoo_per_class_recall.json"

EPOCH_LIST = [0, 4, 8]  # the only iterations with valid acc_class_* values
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load pretrained SANE autoencoder
# ---------------------------------------------------------------------------
logging.info("Loading pretrained SANE model")
config = json.load((TRIAL_DIR / "params.json").open("r"))
config["device"] = device
config["model::compile"] = False  # no need to compile for inference

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
# Linear (closed-form ridge) prediction head, one output per class
# ---------------------------------------------------------------------------
logging.info("Running downstream per-class recall regression")
dstk = DownstreamTaskLearner()
performance = dstk.eval_dstasks(
    model=module,
    trainset=ds_train,
    testset=ds_test,
    valset=ds_val,
    task_keys=ACC_CLASS_KEYS,
    batch_size=256,
)
performance["method"] = "sane_mnist_smallcnnzoo"

# mean test R^2 across classes, for convenience
test_r2 = [performance[f"{k}_test"] for k in ACC_CLASS_KEYS]
performance["mean_acc_class_test"] = sum(test_r2) / len(test_r2)

logging.info(f"Per-class test R^2: {dict(zip(ACC_CLASS_KEYS, test_r2))}")
logging.info(f"Mean test R^2: {performance['mean_acc_class_test']:.4f}")

with open(RESULTS_JSON, "w") as f:
    json.dump(performance, f, indent=4)
logging.info(f"Wrote results to {RESULTS_JSON}")

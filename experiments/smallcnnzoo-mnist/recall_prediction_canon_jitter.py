"""Quantify canonicalization jitter in the per-class recall R^2.

git-re-basin weight_matching visits permutation blocks in torch.randperm order
inside unseeded ray workers, so the canonical form (and hence the embeddings)
varies between dataset builds. This script repeats the full epoch-{8} pipeline
REPS times with the *identical* seed-42 train/val/test split — the spread across
reps is therefore pure canonicalization jitter, to be compared against the
split-variance spread from recall_prediction_r2_spread.py.

Run via recall_prediction_canon_jitter.sh (sbatch).
"""

import logging
import os

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import gc
import json
from pathlib import Path

import torch

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec
from SANE.models.def_AE_module import AEModule
from SANE.models.def_downstream_module import DownstreamTaskLearner

logging.basicConfig(level=logging.INFO)

TRIAL_DIR = Path(
    "sane_pretraining/sane_mnist_smallcnnzoo/"
    "AE_trainable_e550b_00000_0_2026-06-24_17-46-42"
)
CHECKPOINT = TRIAL_DIR / "checkpoint_000010" / "state.pt"
ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist/")
OUT_DIR = Path("recall_prediction/r2_spread")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_JSON = OUT_DIR / "canon_jitter.json"

EPOCH_LIST = [8]
REPS = 3
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.info("Loading pretrained SANE model")
config = json.load((TRIAL_DIR / "params.json").open("r"))
config["device"] = device
config["model::compile"] = False
config["training::steps_per_epoch"] = 1
module = AEModule(config)

checkpoint = torch.load(CHECKPOINT, map_location=device)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
module.model.load_state_dict(state_dict)
module.model.eval()

property_keys = {
    "result_keys": ACC_CLASS_KEYS + ["test_acc", "training_iteration"],
    "config_keys": [],
}


def build_split(split):
    return DatasetTokens(
        root=ZOO_ROOT,
        epoch_lst=EPOCH_LIST,
        mode="vector",
        permutation_spec=smallcnnzoo_permutation_spec(),
        map_to_canonical=True,
        standardize=True,
        tokensize=config["ae:i_dim"],
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


dstk = DownstreamTaskLearner()

results = {"target_keys": ACC_CLASS_KEYS, "epoch_list": EPOCH_LIST, "reps": {}}

for rep in range(REPS):
    logging.info(f"=== canonicalization-jitter rep {rep} ===")
    ds_train = build_split("train")
    ds_val = build_split("val")
    ds_test = build_split("test")
    res = dstk.eval_multivariate_regression(
        model=module,
        trainset=ds_train,
        testset=ds_test,
        valset=ds_val,
        target_keys=ACC_CLASS_KEYS,
        batch_size=256,
    )
    r2_test = [r.item() for r in res["r2_test"]]
    results["reps"][str(rep)] = {
        "reg_best": res["reg_best"],
        "r2_test": dict(zip(ACC_CLASS_KEYS, r2_test)),
        "mean_r2_test": res["mean_r2_test"],
    }
    logging.info(f"rep {rep}: mean test R^2 {res['mean_r2_test']:.4f}")

    r2_mat = torch.tensor(
        [
            [results["reps"][str(r)]["r2_test"][k] for k in ACC_CLASS_KEYS]
            for r in range(rep + 1)
        ]
    )
    results["r2_test_mean"] = dict(zip(ACC_CLASS_KEYS, r2_mat.mean(dim=0).tolist()))
    if rep > 0:
        results["r2_test_std"] = dict(zip(ACC_CLASS_KEYS, r2_mat.std(dim=0).tolist()))
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Wrote rep {rep} to {RESULTS_JSON}")

    del ds_train, ds_val, ds_test
    gc.collect()

logging.info("Done")

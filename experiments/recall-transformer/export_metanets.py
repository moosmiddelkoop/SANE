"""Copy the seed-0 recall transformers of one unthi zoo to the shared metanets directory, as self-contained bundles.

Writes to /projects/prjs2156/shared/wsl/metanets/recall_transformer/<zoo>/:
  preprocessing.pt: what prepare_tokens (def_recall_transformer.py) needs to turn a raw checkpoint
    into tokens: reference_checkpoint (canonical form), layers (per-layer mean and std of the
    train split, checkpoints 0-8), positions, tokensize, split_id. These come from DatasetTokens
    on the train split with the settings of build_tokens.py; the token cache does not store them.
  <zoo>-<init>-seed0.pt: state_dict, model_kwargs (for RecallTransformer), Y_mean and Y_std (the
    model predicts (recall - Y_mean) / Y_std), class_keys, split_id.
  <zoo>-<init>-seed0.json: the metrics of the run (copy of results/<run>.json).
Checks: (1) the bundle reproduces the test MSE at checkpoint 8 of results/<run>.json on the cached
tokens; (2) prepare_tokens on raw train checkpoints gives about the cached tokens and predictions
(logged; canonical form is stochastic, so they are not identical).
--noise, --permutations: export the runs trained with this augmentation (train_recall_transformer.py).
  If preprocessing.pt already exists, it is reused and not rebuilt: canonical form is stochastic,
  so a rebuild would change the input of the bundles already exported.
Run: sbatch export_metanets.sh <zoo> [--noise ...] [--permutations ...] (CPU, about 1-2 h; less with an existing preprocessing.pt).
"""

import os

NUM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

import json
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec
from SANE.models.def_recall_transformer import RecallTransformer, prepare_tokens
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--zoo", choices=["mnist", "fmnist", "cifar10", "svhn"], required=True)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--permutations", type=int, default=0)
args = parser.parse_args()
ZOO = args.zoo
AUGMENTATION = f"-noise{args.noise}-perm{args.permutations}" if args.noise or args.permutations else ""  # as in train_recall_transformer.py

# as in train_recall_transformer.py
PROJECT_DIR = Path("/projects/prjs2156/shared/wsl/unthi_zoo")
ZOO_ROOT = PROJECT_DIR / "unthi_fmnist" if ZOO == "fmnist" else Path(f"/scratch-shared/mmiddelkoop/unthi_zoo/unthi_{ZOO}")
CACHE_PT = PROJECT_DIR / f"unthi_{ZOO}_preprocessed" / "tokens_ep4-8.pt"
SANE_RUNS = {
    "mnist": "mnist-v2.0_6133f_00000_2026-07-15_11-48-49",
    "fmnist": "fmnist-v1.0_d3e0b_00000_2026-07-15_10-54-45",
    "cifar10": "cifar10-v1.0_d496b_00000_2026-07-15_10-54-47",
    "svhn": "svhn-v1.0_d4950_00000_2026-07-15_10-54-47",
}
SANE_CONFIG = json.load((Path("/projects/prjs2156/shared/wsl/metanets/sane_pretraining") / ZOO / SANE_RUNS[ZOO] / "params.json").open())
MODEL_SIZE = {
    "scratch": {"d_model": 256, "nhead": 4, "num_layers": 4},
    "sane": {k: SANE_CONFIG[f"ae:{k}"] for k in ["d_model", "nhead", "num_layers"]},
}
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]
RESULTS_DIR = Path(__file__).parent / "results"
OUT_DIR = Path("/projects/prjs2156/shared/wsl/metanets/recall_transformer") / ZOO
N_RAW_CHECKS = 20

seed_everything(0)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# same settings as build_tokens.py
PREPROCESSING_PT = OUT_DIR / "preprocessing.pt"
if PREPROCESSING_PT.exists():
    preprocessing = torch.load(PREPROCESSING_PT)
else:
    ds = DatasetTokens(
        root=ZOO_ROOT,
        epoch_lst=list(range(9)),
        mode="vector",
        permutation_spec=smallcnnzoo_permutation_spec(),
        map_to_canonical=True,
        standardize=True,
        tokensize=145,
        train_val_test="train",
        weight_threshold=100,
        max_samples=None,
        property_keys={"result_keys": ACC_CLASS_KEYS, "config_keys": []},
        num_threads=NUM_THREADS,
        verbosity=3,
        getitem="tokens+props",
        ignore_bn=True,
    )
    preprocessing = {
        "reference_checkpoint": ds.reference_checkpoint,
        "layers": ds.layers,
        "positions": ds.positions,
        "tokensize": 145,
        "ignore_bn": True,
        "weight_threshold": 100,
        "split_id": ds.split_id,
    }
    del ds
    torch.save(preprocessing, PREPROCESSING_PT)

cache = torch.load(CACHE_PT)
assert cache["split_id"] == preprocessing["split_id"]
assert torch.equal(cache["positions"], preprocessing["positions"])
is_train = torch.tensor([split == "train" for split in cache["split"]])
is_test_ep8 = torch.tensor([split == "test" for split in cache["split"]]) & (cache["epoch"] == 8)
Y_train = cache["Y"][is_train].float()
Y_mean, Y_std = Y_train.mean(dim=0), Y_train.std(dim=0)  # as in train_recall_transformer.py

# raw epoch-8 checkpoints of the first train models, for check 2
raw_check_rows = torch.nonzero(is_train & (cache["epoch"] == 8)).flatten()[:N_RAW_CHECKS]
raw_tokens = torch.stack([
    prepare_tokens(
        torch.load(ZOO_ROOT / cache["model_dir"][row] / "checkpoint_000008" / "checkpoints", map_location="cpu"),
        preprocessing,
        smallcnnzoo_permutation_spec(),
    )
    for row in raw_check_rows
])
logging.info(f"check 2: max |raw tokens - cached tokens| = {(raw_tokens - cache['tokens'][raw_check_rows]).abs().max():.2e}")


@torch.no_grad()
def predict_recall(model, tokens):
    positions = preprocessing["positions"].expand(len(tokens), -1, -1)
    return torch.cat([model(t, p) for t, p in zip(tokens.split(256), positions.split(256))]) * Y_std + Y_mean


for init in ["scratch", "sane"]:
    run = f"{ZOO}-{init}{AUGMENTATION}-seed0"
    model_kwargs = {"i_dim": 145, "max_positions": SANE_CONFIG["ae:max_positions"], "n_classes": 10, "dropout": 0.1, **MODEL_SIZE[init]}
    model = RecallTransformer(**model_kwargs)
    model.load_state_dict(torch.load(RESULTS_DIR / f"{run}.pt", map_location="cpu"))
    model.eval()

    test_mse = ((predict_recall(model, cache["tokens"][is_test_ep8].float()) - cache["Y"][is_test_ep8]) ** 2).mean().item()
    results_mse = json.loads((RESULTS_DIR / f"{run}.json").read_text())["metrics"]["8"]["test"]["mse"]
    logging.info(f"{run} check 1: test MSE at checkpoint 8 = {test_mse:.6f}, results json = {results_mse:.6f}")
    assert abs(test_mse - results_mse) < 0.02 * results_mse  # training evaluated in bf16

    prediction_difference = predict_recall(model, raw_tokens) - predict_recall(model, cache["tokens"][raw_check_rows].float())
    logging.info(f"{run} check 2: max |recall from raw - recall from cache| = {prediction_difference.abs().max():.4f}")

    bundle = {
        "state_dict": model.state_dict(),
        "model_kwargs": model_kwargs,
        "Y_mean": Y_mean,
        "Y_std": Y_std,
        "class_keys": ACC_CLASS_KEYS,
        "split_id": cache["split_id"],
    }
    torch.save(bundle, OUT_DIR / f"{run}.pt")
    shutil.copy(RESULTS_DIR / f"{run}.json", OUT_DIR / f"{run}.json")
    logging.info(f"saved {OUT_DIR / run}.pt")

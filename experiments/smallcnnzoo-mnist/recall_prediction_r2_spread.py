"""R^2 spread of per-class recall prediction over 10 random train/val/test splits,
for epoch sets [0, 4, 8], [4, 8] and [8].

A plain rerun of property_prediction_mnist_smallcnnzoo_multivariate.py is
deterministic (the path shuffle in dataset_epochs.py is seeded with 42), so the
spread must come from varying the train/val/test split. The only thing a
different shuffle seed changes downstream is which *models* land in which
split, so instead of rebuilding the (expensive) DatasetTokens 10 times we:

  1. build + embed each epoch set once (identical pipeline / params to
     property_prediction_mnist_smallcnnzoo_multivariate.py),
  2. pool the embeddings of all three splits with a per-sample model id,
  3. resplit 10 times by model (70/15/15, seeds 0..9) and refit the
     multivariate ridge head per split.

Run via recall_prediction_r2_spread.sh (sbatch).
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
import random
from pathlib import Path

import torch
from einops import repeat

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec
from SANE.models.def_AE_module import AEModule
from SANE.models.def_downstream_module import DownstreamTaskLearner

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths / config (identical to property_prediction_mnist_smallcnnzoo_multivariate.py)
# ---------------------------------------------------------------------------
TRIAL_DIR = Path(
    "sane_pretraining/sane_mnist_smallcnnzoo/"
    "AE_trainable_e550b_00000_0_2026-06-24_17-46-42"
)
CHECKPOINT = TRIAL_DIR / "checkpoint_000010" / "state.pt"
ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist/")
OUT_DIR = Path("recall_prediction/r2_spread")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_JSON = OUT_DIR / "r2_spread.json"

EPOCH_SETS = [[0, 4, 8], [4, 8], [8]]
N_SEEDS = 10
DS_SPLIT = [0.7, 0.15, 0.15]
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]
SENTINEL = -999.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Load pretrained SANE autoencoder
# ---------------------------------------------------------------------------
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

dstk = DownstreamTaskLearner()
dstk.polar_coordinates = False
dstk.device = torch.device(device)


def build_split(epoch_list, split):
    logging.info(f"Building DatasetTokens epochs={epoch_list} split={split}")
    return DatasetTokens(
        root=ZOO_ROOT,
        epoch_lst=epoch_list,
        mode="vector",
        permutation_spec=smallcnnzoo_permutation_spec(),
        map_to_canonical=True,
        standardize=True,
        tokensize=config["ae:i_dim"],
        train_val_test=split,
        ds_split=DS_SPLIT,
        weight_threshold=100,
        max_samples=None,
        property_keys=property_keys,
        shuffle_path=True,
        num_threads=12,
        verbosity=3,
        getitem="tokens+props",
        ignore_bn=True,
    )


def embed_and_targets(ds):
    """embeddings [N, D], targets [N, 10] (sentinel -> NaN), model id per sample [N]"""
    w, _ = ds.__get_weights__()
    try:
        pos = torch.stack(ds.pos)
    except Exception:
        pos = repeat(ds.positions, "n d -> b n d", b=w.shape[0])
    z = dstk.map_embeddings(weights=w, pos=pos, model=module, batch_size=256)
    Y = dstk._stack_target_columns(ds, ACC_CLASS_KEYS)
    Y = torch.where(Y == SENTINEL, torch.full_like(Y, float("nan")), Y)
    mid = torch.tensor(
        [idx for idx in range(len(ds.data)) for _ in range(len(ds.data[idx]))]
    )
    return z, Y, mid


results = {"target_keys": ACC_CLASS_KEYS, "ds_split": DS_SPLIT, "epoch_sets": {}}

for epoch_list in EPOCH_SETS:
    label = "-".join(str(e) for e in epoch_list)
    # --- build + embed once ---
    z_all, Y_all, mid_all, offset = [], [], [], 0
    for split in ["train", "val", "test"]:
        ds = build_split(epoch_list, split)
        z, Y, mid = embed_and_targets(ds)
        z_all.append(z)
        Y_all.append(Y)
        mid_all.append(mid + offset)
        offset += len(ds.data)
        del ds
        gc.collect()
    z_all = torch.cat(z_all)
    Y_all = torch.cat(Y_all)
    mid_all = torch.cat(mid_all)
    logging.info(f"epochs {label}: {z_all.shape[0]} samples from {offset} models")

    # --- 10 random resplits by model, refit ridge head each time ---
    seeds = {}
    for seed in range(N_SEEDS):
        models = list(range(offset))
        random.Random(seed).shuffle(models)
        idx1 = int(DS_SPLIT[0] * offset)
        idx2 = idx1 + int(DS_SPLIT[1] * offset)
        masks = [
            torch.isin(mid_all, torch.tensor(chunk))
            for chunk in (models[:idx1], models[idx1:idx2], models[idx2:])
        ]
        res = dstk.compute_closed_form_solution_multivariate(
            z_train=z_all[masks[0]],
            Y_train=Y_all[masks[0]],
            z_test=z_all[masks[2]],
            Y_test=Y_all[masks[2]],
            z_val=z_all[masks[1]],
            Y_val=Y_all[masks[1]],
        )
        r2_test = [r.item() for r in res["r2_test"]]
        seeds[str(seed)] = {
            "reg_best": res["reg_best"],
            "r2_test": dict(zip(ACC_CLASS_KEYS, r2_test)),
            "mean_r2_test": sum(r2_test) / len(r2_test),
        }
        logging.info(f"epochs {label} seed {seed}: mean test R^2 {seeds[str(seed)]['mean_r2_test']:.4f}")

    r2_mat = torch.tensor(
        [[seeds[str(s)]["r2_test"][k] for k in ACC_CLASS_KEYS] for s in range(N_SEEDS)]
    )
    results["epoch_sets"][label] = {
        "epoch_list": epoch_list,
        "seeds": seeds,
        "r2_test_mean": dict(zip(ACC_CLASS_KEYS, r2_mat.mean(dim=0).tolist())),
        "r2_test_std": dict(zip(ACC_CLASS_KEYS, r2_mat.std(dim=0).tolist())),
    }
    # persist after every epoch set so partial results survive a crash
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Wrote results for epochs {label} to {RESULTS_JSON}")

    del z_all, Y_all, mid_all
    gc.collect()

logging.info("Done")

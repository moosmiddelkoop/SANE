"""Quantify the damage of the epoch-23 loss spike in run c898d (see TODO.md).

Encodes the same checkpoints with the pre-spike checkpoint_000020 and the final
checkpoint_000050 and compares:
  * downstream property-prediction R^2 (per-class recall multivariate head +
    scalar ridge for test_acc / training_iteration),
  * the eigenspectrum / effective rank of the embeddings, plus the z_norm /
    z_var statistics tracked as debug metrics during training.

The DatasetTokens splits are built ONCE and reused for both checkpoints:
canonicalization is stochastic (unseeded randperm in git-re-basin workers), so
re-building per checkpoint would confound the comparison with canon jitter.

Run via compare_spike_checkpoints.sh (sbatch).
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
from einops import repeat

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec
from SANE.models.def_AE_module import AEModule
from SANE.models.def_downstream_module import DownstreamTaskLearner

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
TRIAL_DIR = Path(
    "/projects/prjs2156/shared/wsl/metanets/sane_pretraining/sane_mnist_smallcnnzoo/"
    "AE_trainable_c898d_00000_0_2026-07-02_18-12-12"
)
CHECKPOINTS = {
    "pre_spike_ckpt20": TRIAL_DIR / "checkpoint_000020" / "state.pt",
    "final_ckpt50": TRIAL_DIR / "checkpoint_000050" / "state.pt",
}
ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist/")
OUT_DIR = Path("spike_damage")
os.makedirs(OUT_DIR, exist_ok=True)

EPOCH_LIST = [8]
_label = "-".join(str(e) for e in EPOCH_LIST)
RESULTS_JSON = OUT_DIR / f"compare_spike_checkpoints_{_label}.json"
# embeddings + targets, so future epoch-subset / target refits are a ridge solve
# instead of an hour of re-encoding
EMBEDDINGS_PT = OUT_DIR / f"embeddings_{_label}.pt"
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]
SCALAR_KEYS = ["test_acc", "training_iteration"]
SENTINEL = -999.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# SANE autoencoder (weights swapped per checkpoint below)
# ---------------------------------------------------------------------------
logging.info("Initializing SANE model")
config = json.load((TRIAL_DIR / "params.json").open("r"))
config["device"] = device
config["model::compile"] = False
config["training::steps_per_epoch"] = 1
module = AEModule(config)

property_keys = {
    "result_keys": ACC_CLASS_KEYS + SCALAR_KEYS,
    "config_keys": [],
}

dstk = DownstreamTaskLearner()
dstk.polar_coordinates = False
dstk.device = torch.device(device)

# ---------------------------------------------------------------------------
# Build DatasetTokens once, keep weights / positions / targets per split
# ---------------------------------------------------------------------------
splits = {}
for split in ["train", "val", "test"]:
    logging.info(f"Building DatasetTokens epochs={EPOCH_LIST} split={split}")
    ds = DatasetTokens(
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
    w, _ = ds.__get_weights__()
    try:
        pos = torch.stack(ds.pos)
    except Exception:
        pos = repeat(ds.positions, "n d -> b n d", b=w.shape[0])
    Y = dstk._stack_target_columns(ds, ACC_CLASS_KEYS)
    Y = torch.where(Y == SENTINEL, torch.full_like(Y, float("nan")), Y)
    scalars = {
        key: dstk._stack_target_columns(ds, [key]).squeeze(1) for key in SCALAR_KEYS
    }
    splits[split] = {"w": w, "pos": pos, "Y": Y, "scalars": scalars}
    logging.info(f"split {split}: {w.shape[0]} samples")
    del ds
    gc.collect()


def spectrum_stats(z):
    """eigenspectrum of the embedding covariance + the z_norm/z_var debug stats"""
    zc = z - z.mean(dim=0)
    cov = zc.T @ zc / (zc.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov.double()).flip(0).clamp(min=0)
    p = eig / eig.sum()
    erank = torch.exp(-(p * torch.log(p.clamp(min=1e-12))).sum())
    return {
        "z_norm_mean": z.norm(dim=1).mean().item(),
        "z_var_mean": z.var(dim=0).mean().item(),
        "effective_rank": erank.item(),
        "eigenvalues": eig.tolist(),
    }


# ---------------------------------------------------------------------------
# Encode with each checkpoint, fit heads, compute spectra
# ---------------------------------------------------------------------------
results = {
    "trial": TRIAL_DIR.name,
    "epoch_list": EPOCH_LIST,
    "target_keys": ACC_CLASS_KEYS + SCALAR_KEYS,
    "checkpoints": {},
}
z_store = {
    "epoch_list": EPOCH_LIST,
    "targets": {s: {"recall_Y": splits[s]["Y"], **splits[s]["scalars"]} for s in splits},
    "z": {},
}
for name, ckpt_path in CHECKPOINTS.items():
    logging.info(f"=== checkpoint {name}: {ckpt_path} ===")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    module.model.load_state_dict(state_dict)
    module.model.eval()

    z = {
        split: dstk.map_embeddings(
            weights=s["w"], pos=s["pos"], model=module, batch_size=256
        )
        for split, s in splits.items()
    }

    entry = {"spectrum_test": spectrum_stats(z["test"])}

    logging.info("Fit multivariate per-class recall head")
    res = dstk.compute_closed_form_solution_multivariate(
        z_train=z["train"],
        Y_train=splits["train"]["Y"],
        z_test=z["test"],
        Y_test=splits["test"]["Y"],
        z_val=z["val"],
        Y_val=splits["val"]["Y"],
    )
    entry["recall_reg_best"] = res["reg_best"]
    entry["recall_r2_test"] = dict(
        zip(ACC_CLASS_KEYS, [r.item() for r in res["r2_test"]])
    )
    entry["recall_mean_r2_test"] = res["r2_test"].mean().item()

    for key in SCALAR_KEYS:
        # constant targets (e.g. training_iteration with a single-epoch list) make R^2 undefined
        if splits["train"]["scalars"][key].std() < 1e-8:
            logging.info(f"Skip scalar ridge for {key}: zero variance")
            continue
        logging.info(f"Fit scalar ridge for {key}")
        r2_train, r2_test, r2_val = dstk.compute_closed_form_solution(
            z_train=z["train"],
            prop_train=splits["train"]["scalars"][key].tolist(),
            z_test=z["test"],
            prop_test=splits["test"]["scalars"][key].tolist(),
            z_val=z["val"],
            prop_val=splits["val"]["scalars"][key].tolist(),
        )
        entry[f"{key}_r2_test"] = r2_test

    results["checkpoints"][name] = entry
    logging.info(
        f"{name}: recall mean R^2 {entry['recall_mean_r2_test']:.4f}, "
        f"test_acc R^2 {entry['test_acc_r2_test']:.4f}, "
        f"eff. rank {entry['spectrum_test']['effective_rank']:.1f}, "
        f"z_norm {entry['spectrum_test']['z_norm_mean']:.3f}, "
        f"z_var {entry['spectrum_test']['z_var_mean']:.4f}"
    )
    # persist after every checkpoint so partial results survive a crash
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=4)
    z_store["z"][name] = z
    torch.save(z_store, EMBEDDINGS_PT)

    gc.collect()

logging.info(f"Wrote results to {RESULTS_JSON}")
logging.info("Done")

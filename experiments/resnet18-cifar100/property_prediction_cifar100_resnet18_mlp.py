"""Per-class recall heads on SANE embeddings of the ResNet-18 CIFAR-100 zoo.

Encodes checkpoint_000060 of every model with a pretrained encoder (--tokensize selects
the run). Each model's token sequence is cut into consecutive windows of the training
windowsize (1024, no halo), and the model embedding is the mean over all token
embeddings, each token counted once. Embeddings and targets (acc_class_0..99) are
cached in recall_prediction/embeddings_tk<tokensize>.pt; if the cache exists, encoding
is skipped.

Two heads on the zoo's split.json:
- ridge (closed form): fit on train, regularizer chosen on val, scored on test
- MLP (eval_per_class_recall_MLP): fit on train+val, scored on test, one run per seed,
  logged to W&B

--encoder lq replaces SANE with the LayerQuintiles baseline (per-layer mean, variance and
percentiles of the weights) on the same data and heads, as a control.

Run with the sibling .sh: sbatch property_prediction_cifar100_resnet18_mlp.sh <288|576> [sane|lq]
"""

import logging
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import gc
import json
import statistics
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import torch
import wandb

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.datasets.zoo_split import SplitMismatchError, assert_same_split, load_split
from SANE.git_re_basin.git_re_basin import resnet18_permutation_spec
from SANE.models.def_AE_module import AEModule
from SANE.models.def_downstream_module import DownstreamTaskLearner
from SANE.models.downstream_baselines import LayerQuintiles
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--tokensize", type=int, required=True)
parser.add_argument("--encoder", choices=["sane", "lq"], default="sane")
TOKENSIZE = parser.parse_args().tokensize
ENCODER = parser.parse_args().encoder

PRETRAINING_DIR = Path("/projects/prjs2156/shared/wsl/metanets/sane_pretraining/cifar100_resnet18")
# final checkpoint of each run (288 x 50 epochs, 576 x 100 epochs: same number of steps)
CHECKPOINTS = {
    288: PRETRAINING_DIR / "cifar100-resnet18-tk288-v1.0_86ae3_00000_2026-10-08_11-20-38" / "checkpoint_000050",
    576: PRETRAINING_DIR / "cifar100-resnet18-tk576-v1.0_86998_00000_2026-10-08_11-20-38" / "checkpoint_000100",
}
CHECKPOINT = CHECKPOINTS[TOKENSIZE]
ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/cifar100_resnet18_kaiming_uniform_ep60_no_opt")
OUT_DIR = Path("recall_prediction")
OUT_DIR.mkdir(exist_ok=True)
TAG = f"tk{TOKENSIZE}" if ENCODER == "sane" else f"lq_tk{TOKENSIZE}"
EMBEDDINGS_PT = OUT_DIR / f"embeddings_{TAG}.pt"
RESULTS_JSON = OUT_DIR / f"per_class_recall_{TAG}.json"

EPOCH_LIST = [60]  # the zoo only stores checkpoint_000060
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(100)]
SPLITS = ["train", "val", "test"]

SEEDS = [67]
USE_EMBEDDINGS_CACHE = True  # if False, re-encode the zoo
EPOCHS = 200
LR = 1e-3
MLP_BATCH_SIZE = 64
METRIC_KEYS = ["mse_train", "mse_test", "mae_train", "mae_test", "r2_train", "r2_test"]

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(SEEDS[0])
dstk = DownstreamTaskLearner()
dstk.polar_coordinates = False
dstk.device = torch.device(device)


def embed_models(tokens, positions, module, windowsize):
    """Mean token embedding per model: tokens [N, T, tokensize] -> [N, lat_dim].

    The sequence is cut into consecutive windows of the training windowsize (no halo).
    The last window is moved back so that it is full; the tokens it repeats from the
    previous window are dropped, so every token counts once in the mean."""
    sequence_length = tokens.shape[1]
    starts = [min(start, sequence_length - windowsize) for start in range(0, sequence_length, windowsize)]
    tokens_repeated_by_last_window = starts[-2] + windowsize - starts[-1]
    window_positions = torch.stack([positions[s : s + windowsize] for s in starts]).int().to(device)
    embeddings = []
    for model_tokens in tokens:
        windows = torch.stack([model_tokens[s : s + windowsize] for s in starts]).to(device)
        with torch.no_grad(), torch.autocast("cuda"):
            z = module.forward_encoder(windows, window_positions)  # [n_windows, windowsize, lat_dim]
        z = torch.cat([z[:-1].flatten(0, 1), z[-1, tokens_repeated_by_last_window:]])
        assert z.shape[0] == sequence_length
        embeddings.append(z.float().mean(dim=0).cpu())
    return torch.stack(embeddings)


if EMBEDDINGS_PT.exists() and USE_EMBEDDINGS_CACHE:
    logging.info(f"Loading cached embeddings from {EMBEDDINGS_PT}")
    cache = torch.load(EMBEDDINGS_PT, map_location="cpu")
else:
    logging.info(f"Loading pretrained SANE model from {CHECKPOINT}")
    config = json.load((CHECKPOINT.parent / "params.json").open("r"))
    config["device"] = device
    config["model::compile"] = False
    config["training::steps_per_epoch"] = 1
    module = AEModule(config)
    checkpoint = torch.load(CHECKPOINT / "state.pt", map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    module.model.load_state_dict(state_dict)
    module.model.eval()

    z_all, Y_all, model_dirs, model_split = [], [], [], []
    for split in SPLITS:
        logging.info(f"Building DatasetTokens split={split}")
        # must match the preprocessing (data/preprocess_dataset_resnet18_cifar100_tk*.py)
        ds = DatasetTokens(
            root=ZOO_ROOT,
            epoch_lst=EPOCH_LIST,
            mode="vector",
            permutation_spec=resnet18_permutation_spec(),
            map_to_canonical=True,
            standardize=True,
            tokensize=TOKENSIZE,
            train_val_test=split,
            weight_threshold=float("inf"),
            max_samples=None,
            property_keys={"result_keys": ACC_CLASS_KEYS + ["test_acc"], "config_keys": []},
            num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
            verbosity=3,
            getitem="tokens+props",
            ignore_bn=True,
        )
        assert_same_split(config, ds)
        tokens, _ = ds.__get_weights__()
        if ENCODER == "sane":
            z_all.append(embed_models(tokens, ds.positions, module, config["training::windowsize"]))
        else:
            z_all.append(LayerQuintiles()(tokens, ds.positions.unsqueeze(0)).float())
        Y_all.append(dstk._stack_target_columns(ds, ACC_CLASS_KEYS))
        model_dirs.extend(Path(p[0]).name for p in ds.paths)
        model_split.extend([split] * len(ds.data))
        split_id = ds.split_id
        del ds, tokens
        gc.collect()

    cache = {
        "z": torch.cat(z_all),
        "Y": torch.cat(Y_all),
        "model_dirs": model_dirs,  # zoo dir name per row
        "model_split": model_split,  # split.json split per row
        "split_id": split_id,
        "encoder": str(CHECKPOINT) if ENCODER == "sane" else "LayerQuintiles",
    }
    torch.save(cache, EMBEDDINGS_PT)
    logging.info(f"Saved embeddings cache to {EMBEDDINGS_PT}")

# a cache from another split could put test models into the heads' training data
if cache["split_id"] != load_split(ZOO_ROOT).split_id:
    raise SplitMismatchError(
        f"{EMBEDDINGS_PT} was built on split {cache['split_id']}, not on the current "
        f"split of {ZOO_ROOT}. Delete it or set USE_EMBEDDINGS_CACHE = False to re-encode."
    )
z, Y = cache["z"], cache["Y"]
split_of_row = cache["model_split"]
rows = {split: torch.tensor([s == split for s in split_of_row]) for split in SPLITS}
logging.info(f"{z.shape[0]} models: " + ", ".join(f"{s} {int(rows[s].sum())}" for s in SPLITS))

# ---------------------------------------------------------------------------
# Ridge head: train on train, choose the regularizer on val, score on test
# ---------------------------------------------------------------------------
ridge = dstk.compute_closed_form_solution_multivariate(
    z_train=z[rows["train"]],
    Y_train=Y[rows["train"]],
    z_test=z[rows["test"]],
    Y_test=Y[rows["test"]],
    z_val=z[rows["val"]],
    Y_val=Y[rows["val"]],
)
ridge_summary = {
    "reg_best": ridge["reg_best"],
    "r2_train": ridge["r2_train"].mean().item(),
    "r2_val": ridge["r2_val"].mean().item(),
    "r2_test": ridge["r2_test"].mean().item(),
}
logging.info(f"ridge: {ridge_summary}")

# ---------------------------------------------------------------------------
# MLP head: one run per seed; seeds only change the MLP init
# ---------------------------------------------------------------------------
train_rows = rows["train"] | rows["val"]  # the MLP has no validation set
per_seed = {}
for seed in SEEDS:
    seed_everything(seed)
    wandb.init(
        project="sane-per-class-recall-mlp",
        name=f"cifar100_resnet18_{TAG}_mlp_seed{seed}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        dir=OUT_DIR,
        config={
            "seed": seed,
            "split_id": cache["split_id"],
            "encoder": cache["encoder"],
            "tokensize": TOKENSIZE,
            "target_keys": ACC_CLASS_KEYS,
            "epochs": EPOCHS,
            "lr": LR,
            "mlp_batch_size": MLP_BATCH_SIZE,
        },
    )
    result = dstk.eval_per_class_recall_MLP(
        model=None,
        trainset=(z[train_rows], Y[train_rows]),
        testset=(z[rows["test"]], Y[rows["test"]]),
        target_keys=ACC_CLASS_KEYS,
        epochs=EPOCHS,
        lr=LR,
        mlp_batch_size=MLP_BATCH_SIZE,
        log_fn=wandb.log,
    )
    mlp = result.pop("mlp")
    torch.save(mlp.state_dict(), OUT_DIR / f"mlp_head_{TAG}_seed{seed}.pt")
    per_seed[seed] = {k: result[k] for k in METRIC_KEYS + ["final_train_loss"]}
    wandb.log(per_seed[seed])
    wandb.finish()
    logging.info(f"MLP seed {seed}: R^2 test {result['r2_test']:.4f}  MSE test {result['mse_test']:.6f}")

mean = {k: statistics.mean(r[k] for r in per_seed.values()) for k in METRIC_KEYS}
std = {k: statistics.stdev(r[k] for r in per_seed.values()) if len(per_seed) > 1 else 0.0 for k in METRIC_KEYS}

summary = {
    "encoder": cache["encoder"],
    "tokensize": TOKENSIZE,
    "embeddings_source": str(EMBEDDINGS_PT),
    "split_id": cache["split_id"],
    "target_keys": ACC_CLASS_KEYS,
    "ridge": ridge_summary,
    "mlp": {"epochs": EPOCHS, "lr": LR, "mlp_batch_size": MLP_BATCH_SIZE, "seeds": SEEDS, "per_seed": per_seed, "mean": mean, "std": std},
}
with open(RESULTS_JSON, "w") as f:
    json.dump(summary, f, indent=4)
logging.info(f"Wrote results to {RESULTS_JSON}")

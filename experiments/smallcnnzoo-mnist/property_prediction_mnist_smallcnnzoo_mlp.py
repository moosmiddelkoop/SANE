"""Per-class recall MLP regression head for the smallcnnzoo-mnist SANE model.

Encodes the unthi_mnist zoo at epoch 8 with the latest pretrained encoder
(checkpoint_000050 of the latest run) and caches embeddings + targets + model
ids to recall_prediction/mlp/embeddings.pt. If the cache exists, encoding is
skipped. Then trains the 2-hidden-layer MLP head (eval_per_class_recall_MLP)
once per seed in SEEDS, each time on a fresh 0.8/0.2 re-split by model id
(RESHUFFLE), logs each run to Weights & Biases, and writes per-seed and
mean/std MSE / MAE / R^2 to a JSON file.

Run from this directory with the project venv:
    source "$HOME/SANE/.venv/bin/activate"
    python property_prediction_mnist_smallcnnzoo_mlp.py
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
import random
import statistics
from datetime import datetime
from pathlib import Path

import torch
import wandb
from einops import repeat

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec
from SANE.models.def_AE_module import AEModule
from SANE.models.def_downstream_module import DownstreamTaskLearner
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ZOO = "mnist"
TRIAL_DIR = Path(
    "/projects/prjs2156/shared/wsl/metanets/sane_pretraining/mnist/"
    "mnist-v2.0_6133f_00000_2026-07-15_11-48-49"
)
CHECKPOINT = TRIAL_DIR / "checkpoint_000050" / "state.pt"
ZOO_ROOT = Path("/gpfs/scratch1/shared/mmiddelkoop/unthi_zoo/unthi_mnist/")
OUT_DIR = Path("recall_prediction/mlp")
os.makedirs(OUT_DIR, exist_ok=True)
EMBEDDINGS_PT = OUT_DIR / "embeddings_sorted.pt"
RESULTS_JSON = OUT_DIR / f"mlp_{ZOO}_smallcnnzoo_per_class_recall_10runs_sorted_embeddings.json"

EPOCH_LIST = [8]  # which epoch of the model zoo models to use
EPOCH_SET = "8"  # key in the embeddings cache
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]
DS_SPLIT = [0.8, 0.2]
SPLITS = ["train", "test"]
SENTINEL = -999.0
RESHUFFLE = False  # shuffle model ids per seed before the re-split
# SEEDS = list(range(10))
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
USE_EMBEDDINGS_CACHE = True  # if False, re-encode the zoo (slow)

# MLP training hyperparameters
EPOCHS = 100
LR = 1e-3
MLP_BATCH_SIZE = 64
METRIC_KEYS = ["mse_train", "mse_test", "mae_train", "mae_test", "r2_train", "r2_test"]

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(SEEDS[0])

dstk = DownstreamTaskLearner()
dstk.polar_coordinates = False
dstk.device = torch.device(device)

# ---------------------------------------------------------------------------
# Encode the zoo (or load the cached embeddings)
# ---------------------------------------------------------------------------
if EMBEDDINGS_PT.exists() and USE_EMBEDDINGS_CACHE:
    logging.info(f"Loading cached embeddings from {EMBEDDINGS_PT} (epoch set {EPOCH_SET})")
    cache = torch.load(EMBEDDINGS_PT, map_location="cpu")[EPOCH_SET]
else:
    logging.info("Loading pretrained SANE model")
    config = json.load((TRIAL_DIR / "params.json").open("r"))
    config["seed"] = SEEDS[0]
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
        logging.info(f"Building DatasetTokens split={split}")
        return DatasetTokens(
            root=ZOO_ROOT,
            epoch_lst=EPOCH_LIST,
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
            shuffle_path=False,  # first 0.8xn_models are train, last 0.2xn_models are test
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

    z_all, Y_all, mid_all, dirs_all, offset = [], [], [], [], 0
    for split in SPLITS:
        ds = build_split(split)
        z, Y, mid = embed_and_targets(ds)
        z_all.append(z)
        Y_all.append(Y)
        mid_all.append(mid + offset)
        dirs_all.extend(Path(p[0]).name for p in ds.paths)
        offset += len(ds.data)
        del ds
        gc.collect()

    cache = {
        "z": torch.cat(z_all),
        "Y": torch.cat(Y_all),
        "mid": torch.cat(mid_all),  # model id
        "model_dirs": dirs_all,  # zoo dir name per model id
    }
    torch.save({EPOCH_SET: cache}, EMBEDDINGS_PT)
    logging.info(f"Saved embeddings cache to {EMBEDDINGS_PT}")

z, Y, mid = cache["z"], cache["Y"], cache["mid"]
n_models = int(mid.max()) + 1
logging.info(f"{z.shape[0]} samples, {n_models} models")

# ---------------------------------------------------------------------------
# One MLP run per seed, re-split by model id each time
# ---------------------------------------------------------------------------
per_seed = {}
for seed in SEEDS:
    seed_everything(seed)
    models = list(range(n_models))
    if RESHUFFLE:
        random.Random(seed).shuffle(models)
    idx1 = int(DS_SPLIT[0] * n_models)
    train_mask = torch.isin(mid, torch.tensor(models[:idx1]))
    test_mask = torch.isin(mid, torch.tensor(models[idx1:]))

    wandb.init(
        project="sane-per-class-recall-mlp",
        name=f"{ZOO}_smallcnnzoo_mlp_seed{seed}_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        dir=OUT_DIR,
        config={
            "seed": seed,
            "re_shuffle": RESHUFFLE,
            "encoder": str(CHECKPOINT),
            "embeddings_source": str(EMBEDDINGS_PT),
            "epoch_set": EPOCH_SET,
            "target_keys": ACC_CLASS_KEYS,
            "ds_split": DS_SPLIT,
            "epochs": EPOCHS,
            "lr": LR,
            "mlp_batch_size": MLP_BATCH_SIZE,
            "hidden_dim": 128,
            "n_hidden": 2,
        },
    )
    logging.info(f"Training MLP per-class recall head, seed {seed}")
    result = dstk.eval_per_class_recall_MLP(
        model=None, # type: ignore
        trainset=(z[train_mask], Y[train_mask]),
        testset=(z[test_mask], Y[test_mask]),
        target_keys=ACC_CLASS_KEYS,
        epochs=EPOCHS,
        lr=LR,
        mlp_batch_size=MLP_BATCH_SIZE,
        log_fn=wandb.log,
    )
    mlp = result.pop("mlp")
    torch.save(mlp.state_dict(), OUT_DIR / f"mlp_head_seed{seed}.pt")
    per_seed[seed] = {k: result[k] for k in METRIC_KEYS + ["final_train_loss"]}
    wandb.log(per_seed[seed])
    wandb.finish()
    logging.info(
        f"seed {seed}: MSE test {result['mse_test']:.6f}  "
        f"MAE test {result['mae_test']:.6f}  R^2 test {result['r2_test']:.4f}"
    )

mean = {k: statistics.mean(r[k] for r in per_seed.values()) for k in METRIC_KEYS}
std = {
    k: statistics.stdev(r[k] for r in per_seed.values()) if len(per_seed) > 1 else 0.0
    for k in METRIC_KEYS
}
for k in METRIC_KEYS:
    logging.info(f"{k}: {mean[k]:.6f} +/- {std[k]:.6f}")

# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------
summary = {
    "method": f"sane_{ZOO}_smallcnnzoo_mlp",
    "encoder": str(CHECKPOINT),
    "embeddings_source": str(EMBEDDINGS_PT),
    "epoch_set": EPOCH_SET,
    "epochs": EPOCHS,
    "lr": LR,
    "mlp_batch_size": MLP_BATCH_SIZE,
    "mlp_layer_dims": [m.in_features for m in mlp if isinstance(m, torch.nn.Linear)]
    + [mlp[-1].out_features],
    "mlp_activation": "ReLU",
    "mlp_optimizer": "Adam",
    "re_shuffle": RESHUFFLE,
    "seeds": SEEDS,
    "per_seed": per_seed,
    "mean": mean,
    "std": std,
    "target_keys": ACC_CLASS_KEYS,
}
with open(RESULTS_JSON, "w") as f:
    json.dump(summary, f, indent=4)
logging.info(f"Wrote results to {RESULTS_JSON}")

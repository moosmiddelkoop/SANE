"""Per-class recall MLP regression head for the smallcnnzoo-svhn SANE model.

Encodes the unthi_svhn zoo at epoch 8 with the latest pretrained encoder
(checkpoint_000050 of the latest run) and caches embeddings + targets + model
ids to recall_prediction/mlp/embeddings.pt. If the cache exists, encoding is
skipped. Then trains the 2-hidden-layer MLP head (eval_per_class_recall_MLP)
on a 0.7/0.15/0.15 re-split by model id and logs to Weights & Biases.

Run from this directory with the project venv:
    source "$HOME/SANE/.venv/bin/activate"
    python property_prediction_svhn_smallcnnzoo_mlp.py
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
# Seed
# ---------------------------------------------------------------------------
SEED = 67
seed_everything(SEED)

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
ZOO = "svhn"
TRIAL_DIR = Path(
    "/projects/prjs2156/shared/wsl/metanets/sane_pretraining/svhn/"
    "svhn-v1.0_d4950_00000_2026-07-15_10-54-47"
)
CHECKPOINT = TRIAL_DIR / "checkpoint_000050" / "state.pt"
ZOO_ROOT = Path("/gpfs/scratch1/shared/mmiddelkoop/unthi_zoo/unthi_svhn/")
OUT_DIR = Path("recall_prediction/mlp")
os.makedirs(OUT_DIR, exist_ok=True)
EMBEDDINGS_PT = OUT_DIR / "embeddings.pt"
MLP_PT = OUT_DIR / "mlp_head.pt"
RESULTS_JSON = OUT_DIR / f"mlp_{ZOO}_smallcnnzoo_per_class_recall.json"

EPOCH_LIST = [8]  # which epoch of the model zoo models to use
EPOCH_SET = "8"  # key in the embeddings cache
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]
DS_SPLIT = [0.7, 0.15, 0.15]
SENTINEL = -999.0

# MLP training hyperparameters
EPOCHS = 200
LR = 1e-3
MLP_BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
wandb.init(
    project="sane-per-class-recall-mlp",
    name=f"{ZOO}_smallcnnzoo_mlp" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    dir=OUT_DIR,
    config={
        "seed": SEED,
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

dstk = DownstreamTaskLearner()
dstk.polar_coordinates = False
dstk.device = torch.device(device)

# ---------------------------------------------------------------------------
# Encode the zoo (or load the cached embeddings)
# ---------------------------------------------------------------------------
if EMBEDDINGS_PT.exists():
    logging.info(f"Loading cached embeddings from {EMBEDDINGS_PT} (epoch set {EPOCH_SET})")
    cache = torch.load(EMBEDDINGS_PT, map_location="cpu")[EPOCH_SET]
else:
    logging.info("Loading pretrained SANE model")
    config = json.load((TRIAL_DIR / "params.json").open("r"))
    config["seed"] = SEED
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

    z_all, Y_all, mid_all, offset = [], [], [], 0
    for split in ["train", "val", "test"]:
        ds = build_split(split)
        z, Y, mid = embed_and_targets(ds)
        z_all.append(z)
        Y_all.append(Y)
        mid_all.append(mid + offset)
        offset += len(ds.data)
        del ds
        gc.collect()
    cache = {
        "z": torch.cat(z_all),
        "Y": torch.cat(Y_all),
        "mid": torch.cat(mid_all),
    }
    torch.save({EPOCH_SET: cache}, EMBEDDINGS_PT)
    logging.info(f"Saved embeddings cache to {EMBEDDINGS_PT}")

z, Y, mid = cache["z"], cache["Y"], cache["mid"]
logging.info(f"{z.shape[0]} samples, {int(mid.max()) + 1} models")

# ---------------------------------------------------------------------------
# Re-split by model id
# ---------------------------------------------------------------------------
n_models = int(mid.max()) + 1
models = list(range(n_models))
random.Random(SEED).shuffle(models)
idx1 = int(DS_SPLIT[0] * n_models)
idx2 = idx1 + int(DS_SPLIT[1] * n_models)
train_mask = torch.isin(mid, torch.tensor(models[:idx1]))
test_mask = torch.isin(mid, torch.tensor(models[idx2:]))

# ---------------------------------------------------------------------------
# Train MLP regression head with wandb logging
# ---------------------------------------------------------------------------
logging.info("Training MLP per-class recall head on SANE embeddings")
result = dstk.eval_per_class_recall_MLP(
    model=None,
    trainset=(z[train_mask], Y[train_mask]),
    testset=(z[test_mask], Y[test_mask]),
    target_keys=ACC_CLASS_KEYS,
    epochs=EPOCHS,
    lr=LR,
    mlp_batch_size=MLP_BATCH_SIZE,
    log_fn=wandb.log,
)

torch.save(result.pop("mlp").state_dict(), MLP_PT)
logging.info(f"Saved MLP head state_dict to {MLP_PT}")

logging.info(f"Final train loss: {result['final_train_loss']:.6f}")
logging.info(f"MSE train: {result['mse_train']:.6f}  MSE test: {result['mse_test']:.6f}")
logging.info(f"Mean R^2 train: {result['r2_train']:.4f}  Mean R^2 test: {result['r2_test']:.4f}")

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
    "final_train_loss": result["final_train_loss"],
    "mse_train": result["mse_train"],
    "mse_test": result["mse_test"],
    "r2_train": result["r2_train"],
    "r2_test": result["r2_test"],
    "target_keys": ACC_CLASS_KEYS,
}
with open(RESULTS_JSON, "w") as f:
    json.dump(summary, f, indent=4)
logging.info(f"Wrote results to {RESULTS_JSON}")

wandb.log({
    "final_train_loss": result["final_train_loss"],
    "mse_train": result["mse_train"],
    "mse_test": result["mse_test"],
    "r2_train": result["r2_train"],
    "r2_test": result["r2_test"],
})
wandb.finish()

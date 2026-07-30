"""Per-class recall MLP regression head on cached SANE embeddings.

Trains a 2-hidden-layer MLP (eval_per_class_recall_MLP) on the embeddings
cached by recall_prediction_mse_spread.py in
recall_prediction/mse_spread/embeddings.pt (clipped encoder
gradient-clip-2.0_7a92c, checkpoint_000050), epoch set [8]. Loading the cache
replaces the hour-long zoo re-encode; models are re-split 0.7/0.15/0.15 by
model id. Logs per-step / per-epoch loss to Weights & Biases.

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

import json
import random
from datetime import datetime
from pathlib import Path

import torch
import wandb

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
EMBEDDINGS_PT = Path("recall_prediction/mse_spread/embeddings.pt")
ENCODER = "gradient-clip-2.0_7a92c_00000_2026-07-07_19-01-21/checkpoint_000050"
EPOCH_SET = "8"  # which epoch of the model zoo models to use
OUT_DIR = Path("recall_prediction/mlp")
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_JSON = OUT_DIR / "mlp_mnist_smallcnnzoo_per_class_recall.json"

ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]
DS_SPLIT = [0.7, 0.15, 0.15]

# MLP training hyperparameters
EPOCHS = 200
LR = 1e-3
MLP_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# wandb
# ---------------------------------------------------------------------------
wandb.init(
    project="sane-per-class-recall-mlp",
    name="mnist_smallcnnzoo_mlp" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    dir=OUT_DIR,
    config={
        "seed": SEED,
        "encoder": ENCODER,
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

# ---------------------------------------------------------------------------
# Load cached embeddings + targets, re-split by model id
# ---------------------------------------------------------------------------
logging.info(f"Loading cached embeddings from {EMBEDDINGS_PT} (epoch set {EPOCH_SET})")
cache = torch.load(EMBEDDINGS_PT, map_location="cpu")[EPOCH_SET]
z, Y, mid = cache["z"], cache["Y"], cache["mid"]
logging.info(f"{z.shape[0]} samples, {int(mid.max()) + 1} models")

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
logging.info("Training MLP per-class recall head on cached SANE embeddings")
dstk = DownstreamTaskLearner()
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

MLP_PT = OUT_DIR / "mlp_head.pt"
torch.save(result.pop("mlp").state_dict(), MLP_PT)
logging.info(f"Saved MLP head state_dict to {MLP_PT}")

logging.info(f"Final train loss: {result['final_train_loss']:.6f}")
logging.info(f"MSE train: {result['mse_train']:.6f}  MSE test: {result['mse_test']:.6f}")
logging.info(f"Mean R^2 train: {result['r2_train']:.4f}  Mean R^2 test: {result['r2_test']:.4f}")

# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------
summary = {
    "method": "sane_mnist_smallcnnzoo_mlp",
    "encoder": ENCODER,
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

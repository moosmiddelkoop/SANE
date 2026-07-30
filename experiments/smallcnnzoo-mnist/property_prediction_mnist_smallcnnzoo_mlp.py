"""Per-class recall MLP regression head for the smallcnnzoo-mnist SANE model.

Same setup as property_prediction_mnist_smallcnnzoo_multivariate.py, but trains a
2-hidden-layer MLP (eval_per_class_recall_MLP) instead of a closed-form ridge head,
and logs per-step / per-epoch loss to Weights & Biases.

All DatasetTokens preprocessing params are kept identical to
data/preprocess_dataset_smallcnnzoo_mnist.py so the embeddings stay on-distribution.

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
from pathlib import Path

import torch
import wandb

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
TRIAL_DIR = Path("sane_pretraining/sane_mnist_smallcnnzoo/AE_trainable_e550b_00000_0_2026-06-24_17-46-42")
OUT_DIR = Path("recall_prediction/mlp")
os.makedirs(OUT_DIR, exist_ok=True)
CHECKPOINT = TRIAL_DIR / "checkpoint_000010" / "state.pt"
ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist/")
RESULTS_JSON = OUT_DIR / "mlp_mnist_smallcnnzoo_per_class_recall.json"

EPOCH_LIST = [8]
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]

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
    name="mnist_smallcnnzoo_mlp",
    dir=OUT_DIR,
    config={
        "seed": SEED,
        "epoch_list": EPOCH_LIST,
        "target_keys": ACC_CLASS_KEYS,
        "epochs": EPOCHS,
        "lr": LR,
        "mlp_batch_size": MLP_BATCH_SIZE,
        "hidden_dim": 128,
        "n_hidden": 2,
    },
)

# ---------------------------------------------------------------------------
# Load pretrained SANE autoencoder
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Build DatasetTokens over the zoo at epochs [8]
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


ds_train = build_split("train")
ds_test = build_split("test")

# ---------------------------------------------------------------------------
# Train MLP regression head with wandb logging
# ---------------------------------------------------------------------------
logging.info("Training MLP per-class recall head on SANE embeddings")
dstk = DownstreamTaskLearner()
result = dstk.eval_per_class_recall_MLP(
    model=module,
    trainset=ds_train,
    testset=ds_test,
    target_keys=ACC_CLASS_KEYS,
    batch_size=256,
    epochs=EPOCHS,
    lr=LR,
    mlp_batch_size=MLP_BATCH_SIZE,
    log_fn=wandb.log,
)

logging.info(f"Final train loss: {result['final_train_loss']:.6f}")
logging.info(f"MSE train: {result['mse_train']:.6f}  MSE test: {result['mse_test']:.6f}")
logging.info(f"Mean R^2 train: {result['r2_train']:.4f}  Mean R^2 test: {result['r2_test']:.4f}")

# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------
summary = {
    "method": "sane_mnist_smallcnnzoo_mlp",
    "epoch_list": EPOCH_LIST,
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

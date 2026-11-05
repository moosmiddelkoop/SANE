"""End-to-end per-class recall prediction from weight tokens, one unthi zoo (mnist, fmnist, cifar10 or svhn).

No autoencoder and no frozen embedding: one transformer encoder (RecallTransformer in
src/SANE/models/def_recall_transformer.py) reads the weight tokens of a checkpoint and
predicts its 10 per-class recalls. It is trained end to end with MSE.

--init scratch: small encoder (d_model 256, 4 layers, 4 heads), random initialization.
--init sane: size of the SANE encoder. Tokenizer, position embeddings and transformer come
  from the pretrained SANE encoder of the zoo (SANE_RUNS, checkpoint_000050); the CLS token
  and the head are new. These encoders were pretrained on an older random split without a
  split_id, so they have seen the weights (not the recall) of some models of this test split.

Data: unthi_<zoo>_preprocessed/tokens_ep4-8.pt from build_tokens.py. Checkpoints 4 and 8 are one
training set. Targets are standardized with the train mean and std of each class.
Training: AdamW, cosine learning rate per step, bf16 autocast on GPU. The epoch with the
lowest val loss is kept. Metrics per checkpoint (4, 8) on train and test in recall units, as
in resnet18-cifar100/recall_prediction_allepochs.py: MSE, MAE, R^2 per class (mean over
classes), R^2 per model (mean over models), interaction R^2.
Logs to W&B project sane-recall-transformer. Writes results/<run>.json and results/<run>.pt.
Run: sbatch train_recall_transformer.sh <zoo> [scratch|sane] [seed]
"""

import copy
import json
import logging
import statistics
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset

from SANE.datasets.zoo_split import load_split
from SANE.models.def_recall_transformer import RecallTransformer
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--zoo", choices=["mnist", "fmnist", "cifar10", "svhn"], required=True)
parser.add_argument("--init", choices=["scratch", "sane"], default="scratch")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

PROJECT_DIR = Path("/projects/prjs2156/shared/wsl/unthi_zoo")
# as in build_tokens.py
ZOO_ROOT = PROJECT_DIR / "unthi_fmnist" if args.zoo == "fmnist" else Path(f"/scratch-shared/mmiddelkoop/unthi_zoo/unthi_{args.zoo}")
CACHE_PT = PROJECT_DIR / f"unthi_{args.zoo}_preprocessed" / "tokens_ep4-8.pt"
# the SANE encoders used downstream (CLAUDE.md), all at checkpoint_000050
SANE_RUNS = {
    "mnist": "mnist-v2.0_6133f_00000_2026-07-15_11-48-49",
    "fmnist": "fmnist-v1.0_d3e0b_00000_2026-07-15_10-54-45",
    "cifar10": "cifar10-v1.0_d496b_00000_2026-07-15_10-54-47",
    "svhn": "svhn-v1.0_d4950_00000_2026-07-15_10-54-47",
}
SANE_RUN = Path("/projects/prjs2156/shared/wsl/metanets/sane_pretraining") / args.zoo / SANE_RUNS[args.zoo]
SANE_CHECKPOINT = SANE_RUN / "checkpoint_000050" / "state.pt"
OUT_DIR = Path("results")
RUN_NAME = f"{args.zoo}-{args.init}-seed{args.seed}"

EPOCHS = 100
BATCH_SIZE = 256
LR = {"scratch": 3e-4, "sane": 1e-4}[args.init]
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
SMALL_MODEL_SIZE = {"d_model": 256, "nhead": 4, "num_layers": 4}
SPLITS = ["train", "val", "test"]

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(args.seed)

cache = torch.load(CACHE_PT)
# a cache from another split could put test models into the training data
assert cache["split_id"] == load_split(ZOO_ROOT).split_id, f"{CACHE_PT} is from another split; rebuild it"
rows = {s: torch.tensor([split == s for split in cache["split"]]) for s in SPLITS}
tokens = {s: cache["tokens"][rows[s]] for s in SPLITS}
Y = {s: cache["Y"][rows[s]].float() for s in SPLITS}
epoch_of_sample = {s: cache["epoch"][rows[s]] for s in SPLITS}
Y_mean, Y_std = Y["train"].mean(dim=0), Y["train"].std(dim=0)
positions = cache["positions"].to(device)  # same for every sample
logging.info(", ".join(f"{s}: {len(Y[s])} samples" for s in SPLITS))

sane_config = json.load((SANE_RUN / "params.json").open())
model_size = SMALL_MODEL_SIZE if args.init == "scratch" else {k: sane_config[f"ae:{k}"] for k in SMALL_MODEL_SIZE}
model = RecallTransformer(
    i_dim=tokens["train"].shape[2],
    max_positions=sane_config["ae:max_positions"],
    n_classes=Y["train"].shape[1],
    dropout=DROPOUT,
    **model_size,
)
if args.init == "sane":
    model.load_sane_encoder(torch.load(SANE_CHECKPOINT, map_location="cpu")["model"])
model.to(device)


def loader(split, shuffle):
    return DataLoader(TensorDataset(tokens[split], (Y[split] - Y_mean) / Y_std), batch_size=BATCH_SIZE, shuffle=shuffle)


def forward(batch_tokens):
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        return model(batch_tokens.to(device), positions.expand(len(batch_tokens), -1, -1)).float()


def predict(split):
    """standardized predictions [samples, classes]"""
    model.eval()
    with torch.no_grad():
        return torch.cat([forward(batch_tokens).cpu() for batch_tokens, _ in loader(split, shuffle=False)])


def double_center(matrix):
    """[models, classes] -> the model x class interaction: remove the model means and the class means"""
    return matrix - matrix.mean(dim=1, keepdim=True) - matrix.mean(dim=0) + matrix.mean()


def recall_metrics(predictions, targets):
    """predictions and targets [models, classes] in recall units (as in recall_prediction_allepochs.py)"""
    predictions, targets = predictions.double(), targets.double()
    errors = predictions - targets
    interaction = double_center(targets)
    r2_interaction = 1 - ((interaction - double_center(predictions)) ** 2).sum() / (interaction**2).sum()
    squared_errors = errors**2
    r2_per_class = 1 - squared_errors.sum(dim=0) / ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    r2_per_model = 1 - squared_errors.sum(dim=1) / ((targets - targets.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    return {
        "mse": squared_errors.mean().item(),
        "mae": errors.abs().mean().item(),
        "r2_per_class": r2_per_class.mean().item(),
        "r2_per_model": r2_per_model.mean().item(),
        "r2_interaction": r2_interaction.item(),
    }


train_loader = loader("train", shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader))
wandb.init(
    project="sane-recall-transformer",
    name=RUN_NAME,
    config={"zoo": args.zoo, "init": args.init, "seed": args.seed, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, **model_size, "split_id": cache["split_id"]},
)
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    train_losses = []
    for batch_tokens, batch_targets in train_loader:
        loss = F.mse_loss(forward(batch_tokens), batch_targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())
    val_loss = F.mse_loss(predict("val"), (Y["val"] - Y_mean) / Y_std).item()
    wandb.log({"epoch": epoch, "train_loss": statistics.mean(train_losses), "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})
    logging.info(f"epoch {epoch}: train_loss {statistics.mean(train_losses):.4f}, val_loss {val_loss:.4f}")
    if val_loss < best_val_loss:
        best_val_loss, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
results = {"run": RUN_NAME, "init": args.init, "seed": args.seed, "best_epoch": best_epoch, "best_val_loss": best_val_loss, "metrics": {}}
for split in ["train", "test"]:
    predictions = predict(split) * Y_std + Y_mean
    for model_epoch in sorted(set(epoch_of_sample[split].tolist())):
        at_epoch = epoch_of_sample[split] == model_epoch
        metrics = recall_metrics(predictions[at_epoch], Y[split][at_epoch])
        results["metrics"].setdefault(model_epoch, {})[split] = metrics
        wandb.summary.update({f"{split}/ep{model_epoch}/{k}": v for k, v in metrics.items()})
logging.info(json.dumps(results, indent=2))

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / f"{RUN_NAME}.json").write_text(json.dumps(results, indent=2))
torch.save(best_state, OUT_DIR / f"{RUN_NAME}.pt")
wandb.finish()

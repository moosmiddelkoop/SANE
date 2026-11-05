"""End-to-end per-class recall prediction from weight tokens, ResNet-18 CIFAR-100 zoo.

The same method as train_recall_transformer.py (unthi zoos, --init scratch): one small
transformer encoder (RecallTransformer, d_model 256, 4 layers, 4 heads, random initialization)
reads the weight tokens of a checkpoint and predicts its 100 per-class recalls, trained end to
end with MSE. There is no SANE encoder.

A checkpoint is 39304 tokens of 288 values. Each JOIN consecutive tokens are joined into one
input token of JOIN * 288 = 4608 values (the zero-padded end makes 2457 input tokens), so one
forward pass sees the whole checkpoint. A joined token gets the layer and channel position of
its first token and its own index as the global position.

Data: tokens_ep5-60.bin + tokens_ep5-60_meta.pt from build_tokens_resnet18.py, read into RAM
at the start (272 GB, so the job needs about 300 GB): random reads from GPFS were too slow
(about 30 MB/s per reader, so one epoch would take hours). Checkpoints 5, 10, ..., 60 are one training
set. Targets are standardized with the train mean and std of each class.
Training: AdamW, cosine learning rate per step, bf16 autocast on GPU. The epoch with the
lowest val loss is kept. Metrics per checkpoint epoch on train and test in recall units:
MSE, MAE, R^2 per class, R^2 per model, interaction R^2 (as in train_recall_transformer.py).
Logs to W&B project sane-recall-transformer. Writes results/<run>.json and results/<run>.pt.
Run: sbatch train_recall_transformer_resnet18.sh [seed]
"""

import copy
import json
import logging
import statistics
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from SANE.datasets.zoo_split import load_split
from SANE.models.def_recall_transformer import RecallTransformer
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cache_dir", default="/scratch-shared/mmiddelkoop/recall_transformer/resnet18_cifar100")
parser.add_argument("--epochs", type=int, default=50)
args = parser.parse_args()

ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform")
CACHE_DIR = Path(args.cache_dir)
OUT_DIR = Path("results")
RUN_NAME = f"resnet18-cifar100-scratch-seed{args.seed}"

JOIN = 16
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 0.01
DROPOUT = 0.1
MODEL_SIZE = {"d_model": 256, "nhead": 4, "num_layers": 4}
SPLITS = ["train", "val", "test"]

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(args.seed)

meta = torch.load(CACHE_DIR / "tokens_ep5-60_meta.pt")
# a cache from another split could put test models into the training data
assert meta["split_id"] == load_split(ZOO_ROOT).split_id, "the token cache is from another split; rebuild it"
n_samples, n_tokens, tokensize = meta["shape"]
logging.info("Reading the token cache into RAM")
tokens = torch.from_numpy(np.fromfile(CACHE_DIR / "tokens_ep5-60.bin", dtype=np.float16)).view(meta["shape"])
logging.info("Token cache read")
rows = {s: torch.tensor([split == s for split in meta["split"]]).nonzero().squeeze(1) for s in SPLITS}
Y = {s: meta["Y"][rows[s]].float() for s in SPLITS}
epoch_of_sample = {s: meta["epoch"][rows[s]] for s in SPLITS}
Y_mean, Y_std = Y["train"].mean(dim=0), Y["train"].std(dim=0)
logging.info(", ".join(f"{s}: {len(Y[s])} samples" for s in SPLITS))

n_padding_tokens = -n_tokens % JOIN
joined_positions = meta["positions"][::JOIN].clone()  # layer and channel of the first token
joined_positions[:, 0] = torch.arange(len(joined_positions))
joined_positions = joined_positions.to(device)

model = RecallTransformer(
    i_dim=JOIN * tokensize,
    max_positions=(joined_positions.max(dim=0).values + 1).tolist(),
    n_classes=Y["train"].shape[1],
    dropout=DROPOUT,
    **MODEL_SIZE,
).to(device)
n_parameters = sum(p.numel() for p in model.parameters())
logging.info(f"{n_parameters} parameters, {len(joined_positions)} input tokens of {JOIN * tokensize} values")


def batches(split, shuffle):
    """(tokens [batch, n_tokens, tokensize] on the device, standardized targets [batch, classes]).

    Each sample is copied from RAM straight into a device buffer: a CPU batch (torch.stack)
    took about 2 s per batch of 1.45 GB, most of an epoch."""
    order = torch.randperm(len(rows[split])) if shuffle else torch.arange(len(rows[split]))
    targets = (Y[split] - Y_mean) / Y_std
    for start in range(0, len(order), BATCH_SIZE):
        index = order[start : start + BATCH_SIZE]
        batch_tokens = torch.empty((len(index), n_tokens, tokensize), dtype=torch.float16, device=device)
        for i, row in enumerate(rows[split][index]):
            batch_tokens[i].copy_(tokens[row])
        yield batch_tokens, targets[index]


def forward(batch_tokens):
    batch_tokens = F.pad(batch_tokens.float(), (0, 0, 0, n_padding_tokens))
    batch_tokens = batch_tokens.view(len(batch_tokens), -1, JOIN * tokensize)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        return model(batch_tokens, joined_positions.expand(len(batch_tokens), -1, -1)).float()


def predict(split):
    """standardized predictions [samples, classes]"""
    model.eval()
    with torch.no_grad():
        return torch.cat([forward(batch_tokens).cpu() for batch_tokens, _ in batches(split, shuffle=False)])


def double_center(matrix):
    """[models, classes] -> the model x class interaction: remove the model means and the class means"""
    return matrix - matrix.mean(dim=1, keepdim=True) - matrix.mean(dim=0) + matrix.mean()


def recall_metrics(predictions, targets):
    """predictions and targets [models, classes] in recall units (as in train_recall_transformer.py)"""
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


n_train_batches = -(-len(rows["train"]) // BATCH_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * n_train_batches)
wandb.init(
    project="sane-recall-transformer",
    name=RUN_NAME,
    config={"zoo": "resnet18-cifar100", "init": "scratch", "seed": args.seed, "epochs": args.epochs, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "join": JOIN, **MODEL_SIZE,
            "n_parameters": n_parameters, "split_id": meta["split_id"]},
)
best_val_loss = float("inf")
for epoch in range(args.epochs):
    model.train()
    train_losses = []
    for batch_tokens, batch_targets in batches("train", shuffle=True):
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
results = {"run": RUN_NAME, "seed": args.seed, "best_epoch": best_epoch, "best_val_loss": best_val_loss, "metrics": {}}
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

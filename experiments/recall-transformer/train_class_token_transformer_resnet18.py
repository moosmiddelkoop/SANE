"""Per-class recall from weight tokens with one token per class, ResNet-18 CIFAR-100 zoo.

The recall transformers of train_recall_transformer_resnet18.py learn no model x class part of
recall (interaction R^2 ~ 0), while a linear fit on fc bias and fc row norm gets 0.10-0.15
(understanding/fc_row_probe_resnet18.py). There, JOIN = 16 puts 8 fc rows (classes) in one input
token, and one CLS token must carry all 100 class outputs. Here (ClassTokenRecallTransformer):
- Class token c = the 2 fc tokens of class c (512 weights + 1 bias in 2 * 288 values).
- --input fc: only the 100 class tokens. --input full: the body (all layers before fc, 39104
  tokens joined by JOIN = 16 into 2444 tokens of 4608 values) goes in front.
- One shared head maps the output at each class token to one value.
- --class_index: each class token gets its own learned embedding (else one shared embedding).
- Target: the model x class interaction of recall: recall minus the mean over the model's classes,
  minus (train class mean - train mean) at the same checkpoint epoch, divided by its train std.
  The predictions get their mean over classes removed, so they have no model-level part.
- --max_epoch: train only on checkpoint epochs 5..max_epoch (val and test keep all epochs).
Score: interaction R^2 (as r2_interaction in train_recall_transformer_resnet18.py, per checkpoint
epoch), averaged over epochs 5..45; at epochs 50..60 the interaction is noise. The training epoch
with the best val score is kept.
Data: the token cache of build_tokens_resnet18.py. --input full reads all of it into RAM (272 GB);
--input fc reads only the fc tokens (1.4 GB). Regularization: --noise, --dropout, --weight_decay,
--permutations (as in train_recall_transformer_resnet18.py).
Logs to W&B project sane-recall-transformer. Writes results/<run>.json and results/<run>.pt.
Run: sbatch train_class_token_transformer_resnet18.sh [seed] [--input ...] [--class_index] [--max_epoch ...] ...
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

from SANE.datasets.dataset_auxiliaries import tokenize_checkpoint, tokens_to_checkpoint
from SANE.datasets.zoo_split import load_split
from SANE.git_re_basin.git_re_basin import apply_permutation, resnet18_permutation_spec
from SANE.models.def_recall_transformer import ClassTokenRecallTransformer
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--cache_dir", default="/scratch-shared/mmiddelkoop/recall_transformer/resnet18_cifar100")
parser.add_argument("--input", choices=["fc", "full"], default="fc")
parser.add_argument("--class_index", action="store_true")
parser.add_argument("--max_epoch", type=int, default=60)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--noise", type=float, default=0.1)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--weight_decay", type=float, default=0.1)
parser.add_argument("--permutations", type=int, default=100)
args = parser.parse_args()

ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform")
CACHE_DIR = Path(args.cache_dir)
OUT_DIR = Path("results")
RUN_NAME = (
    f"resnet18-cifar100-classtoken-{args.input}{'-classindex' if args.class_index else ''}-ep5-{args.max_epoch}"
    f"-noise{args.noise}-do{args.dropout}-wd{args.weight_decay}-perm{args.permutations}-seed{args.seed}"
)

JOIN = 16
BATCH_SIZE = 64
LR = 3e-4
MODEL_SIZE = {"d_model": 256, "nhead": 4, "num_layers": 4}
SPLITS = ["train", "val", "test"]
SCORE_EPOCHS = range(5, 46, 5)

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(args.seed)

meta = torch.load(CACHE_DIR / "tokens_ep5-60_meta.pt")
# a cache from another split could put test models into the training data
assert meta["split_id"] == load_split(ZOO_ROOT).split_id, "the token cache is from another split; rebuild it"
n_samples, n_tokens, tokensize = meta["shape"]
recall, epoch_all = meta["Y"].double(), meta["epoch"]
n_classes = recall.shape[1]
n_body_tokens = int((meta["positions"][:, 1] < meta["positions"][:, 1].max()).sum())  # fc is the last layer
assert n_tokens - n_body_tokens == 2 * n_classes and n_body_tokens % JOIN == 0

logging.info("Reading the token cache into RAM")
first_token = 0 if args.input == "full" else n_body_tokens
cache_file = "tokens_ep5-60.bin" if args.input == "full" else "tokens_ep5-60_fc.bin"  # fc: from extract_fc_tokens_resnet18.py
tokens = torch.from_numpy(np.fromfile(CACHE_DIR / cache_file, dtype=np.float16)).view(n_samples, n_tokens - first_token, tokensize)
logging.info(f"Token cache read: {tuple(tokens.shape)}")

is_split = {s: torch.tensor([split == s for split in meta["split"]]) for s in SPLITS}
rows = {s: is_split[s].nonzero().squeeze(1) for s in SPLITS}
rows["train"] = rows["train"][epoch_all[rows["train"]] <= args.max_epoch]

# target: the interaction. Remove the model mean (each sample) and the train class mean (each epoch).
class_part = torch.zeros_like(recall)
for epoch in epoch_all.unique():
    train_class_mean = recall[(epoch_all == epoch) & is_split["train"]].mean(dim=0)
    class_part[epoch_all == epoch] = train_class_mean - train_class_mean.mean()
interaction = recall - recall.mean(dim=1, keepdim=True) - class_part
target_std = interaction[rows["train"]].std().item()
logging.info(", ".join(f"{s}: {len(rows[s])} samples" for s in SPLITS))

if args.permutations:
    reference = torch.load(ZOO_ROOT / meta["model_dir"][0] / "checkpoint_000060" / "checkpoints", map_location="cpu")
    spec = resnet18_permutation_spec()
    # a checkpoint whose values are their own positions in the flattened token sequence
    _, padding_mask, reference_positions = tokenize_checkpoint(reference, tokensize=tokensize, return_mask=True, ignore_bn=True)
    flat_index = torch.arange(n_tokens * tokensize, dtype=torch.float64).view(n_tokens, tokensize)
    index_checkpoint = tokens_to_checkpoint(flat_index, reference_positions, {k: v.double() for k, v in reference.items()}, ignore_bn=True)
    permutation_pool = []
    for _ in range(args.permutations):
        permutation = {p: torch.randperm(reference[axes[0][0]].shape[axes[0][1]]) for p, axes in spec.perm_to_axes.items()}
        permuted_index, _, _ = tokenize_checkpoint(apply_permutation(spec, permutation, index_checkpoint), tokensize=tokensize, return_mask=True, ignore_bn=True)
        permutation_pool.append(permuted_index.view(-1).to(torch.int64))
    # keep the part of the tokens in RAM. A permutation moves values only within a layer, so the fc
    # part maps into itself. Padding entries point at value 0 of the full sequence; the padding
    # mask sets them back to 0.
    offset = first_token * tokensize
    permutation_pool = (torch.stack(permutation_pool)[:, offset:] - offset).clamp(min=0).to(device=device, dtype=torch.int32)
    padding_mask = padding_mask[first_token:].to(device=device, dtype=torch.float16)
    logging.info(f"{args.permutations} permutations ready")

body_positions = meta["positions"][:n_body_tokens:JOIN].clone()  # layer and channel of the first token
body_positions[:, 0] = torch.arange(len(body_positions))
body_positions = body_positions.to(device)
model = ClassTokenRecallTransformer(
    class_dim=2 * tokensize,
    n_classes=n_classes,
    class_index=args.class_index,
    dropout=args.dropout,
    body_dim=JOIN * tokensize if args.input == "full" else None,
    max_positions=(body_positions.max(dim=0).values + 1).tolist(),
    **MODEL_SIZE,
).to(device)
n_parameters = sum(p.numel() for p in model.parameters())
logging.info(f"{n_parameters} parameters")


def batches(split, shuffle):
    """(tokens [batch, tokens in RAM, tokensize] on the device, standardized targets [batch, classes]).

    Each sample is copied from RAM straight into a device buffer: a CPU batch (torch.stack)
    of the full tokens took about 2 s per batch of 1.45 GB."""
    order = torch.randperm(len(rows[split])) if shuffle else torch.arange(len(rows[split]))
    for start in range(0, len(order), BATCH_SIZE):
        batch_rows = rows[split][order[start : start + BATCH_SIZE]]
        batch_tokens = torch.empty((len(batch_rows), *tokens.shape[1:]), dtype=torch.float16, device=device)
        for i, row in enumerate(batch_rows):
            batch_tokens[i].copy_(tokens[row])
        yield batch_tokens, (interaction[batch_rows] / target_std).float()


def forward(batch_tokens):
    """standardized interaction predictions [batch, classes], mean 0 over the classes"""
    batch_tokens = batch_tokens.float()
    class_tokens = batch_tokens[:, -2 * n_classes :].reshape(len(batch_tokens), n_classes, 2 * tokensize)
    body_tokens = None
    if args.input == "full":
        body_tokens = batch_tokens[:, :n_body_tokens].reshape(len(batch_tokens), -1, JOIN * tokensize)
    with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
        predictions = model(class_tokens, body_tokens, body_positions.expand(len(batch_tokens), -1, -1)).float()
    return predictions - predictions.mean(dim=1, keepdim=True)


def predict(split):
    """interaction predictions [samples, classes] in recall units"""
    model.eval()
    with torch.no_grad():
        return torch.cat([forward(batch_tokens).cpu() for batch_tokens, _ in batches(split, shuffle=False)]).double() * target_std


def double_center(matrix):
    """[models, classes] -> the model x class interaction: remove the model means and the class means"""
    return matrix - matrix.mean(dim=1, keepdim=True) - matrix.mean(dim=0) + matrix.mean()


def r2_interaction_per_epoch(split, predictions):
    """{checkpoint epoch: interaction R^2}, as r2_interaction in train_recall_transformer_resnet18.py"""
    result = {}
    for epoch in sorted(set(epoch_all[rows[split]].tolist())):
        at_epoch = epoch_all[rows[split]] == epoch
        targets = double_center(recall[rows[split]][at_epoch])
        result[epoch] = 1 - ((targets - double_center(predictions[at_epoch])) ** 2).sum().item() / (targets**2).sum().item()
    return result


def score(r2_per_epoch):
    return statistics.mean(r2_per_epoch[epoch] for epoch in SCORE_EPOCHS)


n_train_batches = -(-len(rows["train"]) // BATCH_SIZE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * n_train_batches)
wandb.init(
    project="sane-recall-transformer",
    name=RUN_NAME,
    config={"zoo": "resnet18-cifar100", "model": "class-token", **vars(args), "batch_size": BATCH_SIZE, "lr": LR, "join": JOIN,
            **MODEL_SIZE, "n_parameters": n_parameters, "split_id": meta["split_id"]},
)
best_score = -float("inf")
for epoch in range(args.epochs):
    model.train()
    train_losses = []
    for batch_tokens, batch_targets in batches("train", shuffle=True):
        if args.permutations:
            for i, choice in enumerate(torch.randint(args.permutations, (len(batch_tokens),))):
                batch_tokens[i] = batch_tokens[i].view(-1)[permutation_pool[choice].long()].view(batch_tokens.shape[1:]) * padding_mask
        if args.noise > 0:
            batch_tokens = batch_tokens * (1 + args.noise * torch.randn_like(batch_tokens))
        loss = F.mse_loss(forward(batch_tokens), batch_targets.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())
    val_score = score(r2_interaction_per_epoch("val", predict("val")))
    wandb.log({"epoch": epoch, "train_loss": statistics.mean(train_losses), "val_r2_interaction_ep5-45": val_score, "lr": scheduler.get_last_lr()[0]})
    logging.info(f"epoch {epoch}: train_loss {statistics.mean(train_losses):.4f}, val_r2_interaction_ep5-45 {val_score:.4f}")
    if val_score > best_score:
        best_score, best_epoch, best_state = val_score, epoch, copy.deepcopy(model.state_dict())

model.load_state_dict(best_state)
results = {"run": RUN_NAME, **vars(args), "best_epoch": best_epoch, "best_val_r2_interaction_ep5-45": best_score, "r2_interaction": {}}
for split in SPLITS:
    results["r2_interaction"][split] = r2_interaction_per_epoch(split, predict(split))
    wandb.summary.update({f"{split}/ep{e}/r2_interaction": r2 for e, r2 in results["r2_interaction"][split].items()})
results["test_r2_interaction_ep5-45"] = score(results["r2_interaction"]["test"])
logging.info(json.dumps(results, indent=2))

OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / f"{RUN_NAME}.json").write_text(json.dumps(results, indent=2))
torch.save(best_state, OUT_DIR / f"{RUN_NAME}.pt")
wandb.finish()

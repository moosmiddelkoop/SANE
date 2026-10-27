"""Mean and SD of the train, validation and test accuracy over the models of the
CIFAR-100 ResNet-18 zoo, for every epoch (training_iteration) in result.json.

Reads train_acc, val_acc and test_acc from each model's result.json and writes
zoo_statistics/accuracy_over_epochs.png. Reads the ep60 zoo: its result.json files
hold all 61 epochs and match those of the full zoo (same trial ids), so the plot does
not need the full download.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ZOO_ROOT = Path(
    "/projects/prjs2156/shared/wsl/cifar100_resnet18/cifar100_resnet18_kaiming_uniform_ep60_no_opt"
)
SPLITS = {"train_acc": ("train", "#2a78d6"), "val_acc": ("validation", "#eb6834"), "test_acc": ("test", "#1baf7a")}
OUT_DIR = Path("zoo_statistics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# accuracy[split][epoch] is the list of accuracies over the models
accuracy = {split: {} for split in SPLITS}
model_dirs = sorted(d for d in ZOO_ROOT.iterdir() if d.is_dir())
for model_dir in model_dirs:
    for line in (model_dir / "result.json").read_text().splitlines():
        line = json.loads(line)
        for split in SPLITS:
            accuracy[split].setdefault(line["training_iteration"], []).append(line[split])

fig, ax = plt.subplots(figsize=(7, 4.5))
for split, (label, color) in SPLITS.items():
    epochs = sorted(accuracy[split])
    mean = np.array([np.mean(accuracy[split][epoch]) for epoch in epochs])
    std = np.array([np.std(accuracy[split][epoch]) for epoch in epochs])
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2, lw=0)
    ax.plot(epochs, mean, color=color, lw=2, label=label)
    print(f"{label}: epoch {epochs[-1]} mean {mean[-1]:.4f} SD {std[-1]:.4f}, "
          f"models per epoch {min(len(v) for v in accuracy[split].values())}-{max(len(v) for v in accuracy[split].values())}")

ax.set_xlabel("epoch")
ax.set_ylabel("accuracy (mean ± SD over models)")
ax.set_title(f"CIFAR-100 ResNet-18 zoo, {len(model_dirs)} models")
ax.set_ylim(0, 1)
ax.grid(axis="y", color="#e5e5e5", lw=0.8)
ax.legend(frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT_DIR / "accuracy_over_epochs.png", dpi=150)

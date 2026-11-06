"""Progress figure of the ResNet-18 CIFAR-100 recall transformer runs, as in karpathy/autoresearch.

One point per run: its lowest val loss (standardized per-class recall MSE, best_val_loss in
results/<run>.json). A run is kept if it is lower than every run before it, else discarded.
The line is the running best. Only --target recall runs: the accuracy run has another target,
so its val loss is not comparable. Runs without a results file yet are skipped.
Writes ~/SANE/experiments/recall-transformer/progress_resnet18.png. Run: python plot_progress_resnet18.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path.home() / "SANE/experiments/recall-transformer/results"
# in the order they were run: (results file, what changed)
RUNS = [
    ("resnet18-cifar100-scratch-seed0.json", "baseline"),
    ("resnet18-cifar100-recall-noise0.1-do0.3-wd0.1-perm0-seed0.json", "noise 0.1, dropout 0.3, weight decay 0.1"),
    ("resnet18-cifar100-recall-noise0.1-do0.3-wd0.1-perm100-seed0.json", "+ permutation augmentation (100)"),
]
KEPT_COLOR, DISCARDED_COLOR = "#1baf7a", "#b4b2a9"

runs = [(label, json.loads((RESULTS_DIR / name).read_text())["best_val_loss"]) for name, label in RUNS if (RESULTS_DIR / name).exists()]
fig, ax = plt.subplots(figsize=(10, 5))
best, running_best, n_kept = float("inf"), [], 0
for index, (label, val_loss) in enumerate(runs):
    kept = val_loss < best
    best = min(best, val_loss)
    running_best.append(best)
    n_kept += kept
    if kept:
        ax.scatter(index, val_loss, s=60, color=KEPT_COLOR, edgecolor="black", linewidth=0.6, zorder=3, label="Kept" if n_kept == 1 else None)
        ax.annotate(label, (index, val_loss), xytext=(8, 8), textcoords="offset points", rotation=25, fontsize=9, color="#0f6e56")
    else:
        ax.scatter(index, val_loss, s=25, color=DISCARDED_COLOR, zorder=2, label="Discarded" if index - n_kept == 0 else None)
        ax.annotate(label, (index, val_loss), xytext=(8, -14), textcoords="offset points", fontsize=8, color="#5f5e5a")
ax.step(range(len(runs)), running_best, where="post", color=KEPT_COLOR, linewidth=2, alpha=0.8, label="Running best")
ax.set_title(f"Recall transformer, ResNet-18 CIFAR-100: {len(runs)} runs, {n_kept} kept")
ax.set_xlabel("Run #")
ax.set_ylabel("Lowest val loss (standardized MSE, lower is better)")
ax.set_xticks(range(len(runs)))
ax.set_xlim(-0.5, max(len(runs) - 0.5, 2.5))
ax.grid(alpha=0.3)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(RESULTS_DIR.parent / "progress_resnet18.png", dpi=150)

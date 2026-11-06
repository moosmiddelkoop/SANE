"""Progress figure of the ResNet-18 CIFAR-100 recall runs, as in karpathy/autoresearch.

One point per class-token run (train_class_token_transformer_resnet18.py): its best val
interaction R^2, averaged over checkpoint epochs 5..45 (best_val_r2_interaction_ep5-45). A run is
kept if it is higher than every run before it, else discarded. The line is the running best.
Two reference lines, same score:
- the fc-row probe (understanding/fc_row_probe_resnet18.py, bias + row norm, val);
- the best earlier recall transformer (train_recall_transformer_resnet18.py, one CLS output):
  test, as those runs did not save val interaction R^2.
Runs without a results file yet are skipped.
Writes ~/SANE/experiments/recall-transformer/progress_resnet18.png. Run: python plot_progress_resnet18.py
"""

import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path.home() / "SANE/experiments/recall-transformer/results"
PROBE_RESULTS = Path.home() / "SANE/understanding/fc_row_probe_resnet18.json"
SCORE_EPOCHS = [str(e) for e in range(5, 46, 5)]
SCORE = "best_val_r2_interaction_ep5-45"
# in the order they were run: (results file, what changed)
RUNS = [
    ("resnet18-cifar100-classtoken-fc-ep5-60-noise0.1-do0.3-wd0.1-perm100-seed0.json", "class tokens, fc only"),
    ("resnet18-cifar100-classtoken-full-ep5-60-noise0.1-do0.3-wd0.1-perm100-seed0.json", "class tokens + body"),
    ("resnet18-cifar100-classtoken-fc-classindex-ep5-60-noise0.1-do0.3-wd0.1-perm100-seed0.json", "fc only, class index"),
    ("resnet18-cifar100-classtoken-fc-ep5-45-noise0.1-do0.3-wd0.1-perm100-seed0.json", "fc only, train epochs 5-45"),
]
EARLIER_RUNS = [
    "resnet18-cifar100-scratch-seed0.json",
    "resnet18-cifar100-recall-noise0.1-do0.3-wd0.1-perm0-seed0.json",
    "resnet18-cifar100-recall-noise0.1-do0.3-wd0.1-perm100-seed0.json",
]
KEPT_COLOR, DISCARDED_COLOR, REFERENCE_COLOR = "#1baf7a", "#b4b2a9", "#5f5e5a"

probe = json.loads(PROBE_RESULTS.read_text())
probe_score = statistics.mean(probe[e]["bias+norm"]["r2_interaction_val"] for e in SCORE_EPOCHS)
earlier_score = max(
    statistics.mean(json.loads((RESULTS_DIR / name).read_text())["metrics"][e]["test"]["r2_interaction"] for e in SCORE_EPOCHS)
    for name in EARLIER_RUNS
)
runs = [(label, json.loads((RESULTS_DIR / name).read_text())[SCORE]) for name, label in RUNS if (RESULTS_DIR / name).exists()]

fig, ax = plt.subplots(figsize=(10, 5))
ax.axhline(probe_score, color=REFERENCE_COLOR, linestyle="--", linewidth=1, label=f"fc-row probe, linear ({probe_score:.3f})")
ax.axhline(earlier_score, color=REFERENCE_COLOR, linestyle=":", linewidth=1, label=f"earlier recall transformers, test ({earlier_score:.3f})")
best, running_best, n_kept = -float("inf"), [], 0
for index, (label, score) in enumerate(runs):
    kept = score > best
    best = max(best, score)
    running_best.append(best)
    n_kept += kept
    if kept:
        ax.scatter(index, score, s=60, color=KEPT_COLOR, edgecolor="black", linewidth=0.6, zorder=3, label="Kept" if n_kept == 1 else None)
        ax.annotate(label, (index, score), xytext=(8, 8), textcoords="offset points", rotation=25, fontsize=9, color="#0f6e56")
    else:
        ax.scatter(index, score, s=25, color=DISCARDED_COLOR, zorder=2, label="Discarded" if index - n_kept == 0 else None)
        ax.annotate(label, (index, score), xytext=(8, -14), textcoords="offset points", fontsize=8, color="#5f5e5a")
if runs:
    ax.step(range(len(runs)), running_best, where="post", color=KEPT_COLOR, linewidth=2, alpha=0.8, label="Running best")
ax.set_title(f"Class-token recall transformer, ResNet-18 CIFAR-100: {len(runs)} runs, {n_kept} kept")
ax.set_xlabel("Run #")
ax.set_ylabel("Val interaction R², mean over epochs 5-45 (higher is better)")
ax.set_xticks(range(max(len(runs), 1)))
ax.set_xlim(-0.5, max(len(runs) - 0.5, 3.5))
ax.grid(alpha=0.3)
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(RESULTS_DIR.parent / "progress_resnet18.png", dpi=150)

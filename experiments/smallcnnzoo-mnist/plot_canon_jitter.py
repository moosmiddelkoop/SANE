"""Analyze the canonicalization-jitter experiment.

Two panels over the 10 recall classes (epoch set {8} in both experiments):
  A) absolute test R^2: the 3 end-to-end canon-jitter reps (identical seed-42
     split) vs the split-variance mean +- std (10 random resplits). Shows the
     reps are near-identical and the systematic offset of the resplit protocol.
  B) std per class: canon jitter (3 reps) vs split variance (10 seeds).

Reads recall_prediction/r2_spread/{canon_jitter,r2_spread}.json and writes
canon_jitter.png/.pdf next to them.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("recall_prediction/r2_spread")
canon = json.load((OUT_DIR / "canon_jitter.json").open())
spread = json.load((OUT_DIR / "r2_spread.json").open())["epoch_sets"]["8"]

keys = canon["target_keys"]
x = np.arange(len(keys))

canon_reps = np.array(
    [[canon["reps"][str(r)]["r2_test"][k] for k in keys] for r in range(len(canon["reps"]))]
)
canon_std = np.array([canon["r2_test_std"][k] for k in keys])
split_mean = np.array([spread["r2_test_mean"][k] for k in keys])
split_std = np.array([spread["r2_test_std"][k] for k in keys])

BLUE = "#2a78d6"   # canonicalization jitter (fixed seed-42 split)
RED = "#e34948"    # split variance (10 random resplits)
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

fig, (axa, axb) = plt.subplots(
    2, 1, figsize=(7.5, 6.5), sharex=True, height_ratios=[1.5, 1]
)

# --- panel A: absolute R^2 ---
axa.errorbar(
    x + 0.12, split_mean, yerr=split_std, fmt="o", ms=5.5, color=RED,
    ecolor=RED, elinewidth=1.8, capsize=3,
    label="10 random splits (mean $\\pm$ std)",
)
for r in range(canon_reps.shape[0]):
    axa.plot(
        x - 0.12, canon_reps[r], "o", ms=5.5, color=BLUE, alpha=0.75,
        label="3 canonicalization reps (fixed split)" if r == 0 else None,
    )
axa.set_ylabel("test $R^2$", color=INK)
axa.set_title(
    "Canonicalization jitter vs split variance — per-class recall, epoch set {8}",
    color=INK, fontsize=11,
)
axa.legend(frameon=False, fontsize=9, loc="lower right")

# --- panel B: spread magnitude ---
w = 0.38
axb.bar(x - w / 2, canon_std, width=w - 0.04, color=BLUE,
        label="std over 3 canon reps")
axb.bar(x + w / 2, split_std, width=w - 0.04, color=RED,
        label="std over 10 splits")
axb.set_ylabel("std of test $R^2$", color=INK)
axb.set_xlabel("recall class", color=INK)
axb.legend(frameon=False, fontsize=9, loc="upper right")

for ax in (axa, axb):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
axb.set_xticks(x)
axb.set_xticklabels([str(i) for i in range(len(keys))])

fig.tight_layout()
fig.savefig(OUT_DIR / "canon_jitter.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT_DIR / "canon_jitter.pdf", bbox_inches="tight")
print(f"wrote {OUT_DIR}/canon_jitter.png / .pdf")

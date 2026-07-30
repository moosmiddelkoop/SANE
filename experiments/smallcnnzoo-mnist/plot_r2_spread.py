"""Plot per-class recall-prediction R^2 (mean +- std over 10 random splits)
against the epoch set used for encoding: {0,4,8} -> {4,8} -> {8}.

Reads recall_prediction/r2_spread/r2_spread.json (written by
recall_prediction_r2_spread.py) and writes r2_spread.png/.pdf next to it.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_JSON = Path("recall_prediction/mse_spread/mse_spread.json")
OUT_STEM = RESULTS_JSON.parent / "r2_spread"

# CVD-validated 10-slot categorical palette (classes 0..9)
PALETTE = [
    "#2a78d6", "#eda100", "#1baf7a", "#e34948", "#4a3aa7",
    "#eb6834", "#008300", "#e87ba4", "#a2652c", "#0aa2c0",
]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

results = json.load(RESULTS_JSON.open())
keys = results["target_keys"]
labels = list(results["epoch_sets"].keys())  # insertion order: 0-4-8, 4-8, 8
x = np.arange(len(labels))

mean = np.array(
    [[results["epoch_sets"][l]["r2_test_mean"][k] for l in labels] for k in keys]
)
std = np.array(
    [[results["epoch_sets"][l]["r2_test_std"][k] for l in labels] for k in keys]
)

fig, ax = plt.subplots(figsize=(7.5, 5.5))
for c in range(len(keys)):
    ax.fill_between(x, mean[c] - std[c], mean[c] + std[c], color=PALETTE[c], alpha=0.15, lw=0)
    ax.plot(x, mean[c], color=PALETTE[c], lw=1.8, marker="o", ms=5.5, label=f"class {c}")

# direct labels at the right edge, nudged apart where lines end too close together
ys = mean[:, -1].copy()
order = np.argsort(ys)
ylo, yhi = ax.get_ylim()
min_gap = 0.026 * (yhi - ylo)
for i in range(1, len(order)):  # push up to enforce the gap
    if ys[order[i]] - ys[order[i - 1]] < min_gap:
        ys[order[i]] = ys[order[i - 1]] + min_gap
if ys[order[-1]] > yhi:  # clamp to the axis top, pushing back down
    ys[order[-1]] = yhi
    for i in range(len(order) - 2, -1, -1):
        if ys[order[i + 1]] - ys[order[i]] < min_gap:
            ys[order[i]] = ys[order[i + 1]] - min_gap
for c in range(len(keys)):
    ax.annotate(
        str(c),
        xy=(x[-1], mean[c, -1]),
        xytext=(x[-1] + 0.07, ys[c]),
        color=PALETTE[c],
        fontsize=10,
        fontweight="bold",
        va="center",
    )

ax.set_xticks(x)
ax.set_xticklabels(["{" + l.replace("-", ", ") + "}" for l in labels])
ax.set_xlim(-0.15, len(labels) - 1 + 0.35)
ax.set_xlabel("epoch set used for encoding", color=INK)
ax.set_ylim(0, 1.0)
ax.set_ylabel("test $R^2$ (per-class recall)", color=INK)
ax.set_title(
    "Per-class recall prediction from SANE embeddings\n"
    "mean $\\pm$ std over 10 random 70/15/15 splits",
    color=INK,
    fontsize=11,
)
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(MUTED)
ax.tick_params(colors=MUTED, labelcolor=INK)
ax.legend(
    ncols=5, fontsize=8, frameon=False, loc="lower center",
    bbox_to_anchor=(0.5, -0.26), handlelength=1.4, columnspacing=1.0,
)

fig.tight_layout()
fig.savefig(OUT_STEM.with_suffix(".png"), dpi=200, bbox_inches="tight")
fig.savefig(OUT_STEM.with_suffix(".pdf"), bbox_inches="tight")
print(f"wrote {OUT_STEM}.png / .pdf")

"""Bar charts of the per-class recall heads of recall_prediction_allepochs.py.

One figure per head (ridge, MLP) and scope (one general head, one head per epoch), with
one panel per metric: MSE, MAE, R^2 per class, R^2 per model and the interaction R^2. Each panel has one bar
group per epoch with train, test and the class-mean predictor on test. MLP bars are the
mean over the seeds, with the std as error bars. Writes
recall_prediction/allepochs/<head>_<scope>.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path("recall_prediction/allepochs")
RESULTS_JSON = OUT_DIR / "per_class_recall_cifar100-resnet18-tk288-ep0-60-v1.0_d75a9_00000_2026-10-28_12-08-02.json"
METRICS = {"mse": "MSE", "mae": "MAE", "r2_per_class": "R² per class (mean over classes)", "r2_per_model": "R² per model (mean over models)", "r2_interaction": "Interaction R²×"}
BARS = {"train": ("train", "#2a78d6"), "test": ("test", "#1baf7a"), "class_mean": ("class-mean predictor (test)", "#9a9a96")}
HEADS = {"ridge": "Ridge", "mlp": "MLP"}
SCOPES = {"general": "one head for epochs 5-60", "per_epoch": "one head per epoch"}

summary = json.loads(RESULTS_JSON.read_text())
results = summary["results"]
epochs = [str(epoch) for epoch in summary["head_epochs"]]
x = np.arange(len(epochs))
bar_width = 0.27

for head, head_name in HEADS.items():
    for scope, scope_name in SCOPES.items():
        fig, axes = plt.subplots(1, len(METRICS), figsize=(25, 4.5))
        for ax, (metric, metric_name) in zip(axes, METRICS.items()):
            for bar_index, (bar, (label, color)) in enumerate(BARS.items()):
                if bar == "class_mean":
                    values, errors = [results["class_mean"][e]["test"][metric] for e in epochs], None
                else:
                    scores = [results[head][scope]["per_epoch"][e][bar][metric] for e in epochs]
                    values = [s["mean"] if head == "mlp" else s for s in scores]
                    errors = [s["std"] for s in scores] if head == "mlp" else None
                ax.bar(x + (bar_index - 1) * bar_width, values, bar_width - 0.02, yerr=errors, color=color, label=label, capsize=2)
            if metric.startswith("r2"):
                ax.axhline(0, color="#444441", lw=0.8)
            ax.set_xticks(x, epochs)
            ax.set_xlabel("zoo epoch")
            ax.set_title(metric_name)
            ax.grid(axis="y", color="#e4e4e0", lw=0.6)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(BARS), frameon=False)
        fig.suptitle(f"{head_name}, {scope_name}: per-class recall from SANE embeddings (tk288, ep0-60 encoder)")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        out_png = OUT_DIR / f"{head}_{scope}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"wrote {out_png}")

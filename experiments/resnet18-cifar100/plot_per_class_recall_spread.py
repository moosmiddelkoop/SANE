"""Spread of per-class recall between the models of the full CIFAR-100 ResNet-18 zoo.

One figure per epoch in EPOCHS: the 100 classes sorted by median recall over the
models, the median as a line and the distribution over the models as a violin per
class. The dashed line is the median test accuracy. The 3 best and 3 worst classes
are labelled. The y-axis is the same for every epoch, so the figures compare.

A second figure per epoch: the Pearson correlation of the class recalls between
the models, with the classes in the same order. The null is the 5-95% range of the
off-diagonal r after each class column is shuffled over the models independently.

Reads acc_class_0..99 from each model's result.json (written by
evaluate_per_class_recall_cifar100_resnet18.py) and writes
recall_prediction/recall_spread/per_class_recall_epoch{EE}.png and
per_class_recall_correlation_epoch{EE}.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

ZOO_ROOT = Path(
    "/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform"
)
CIFAR100_ROOT = Path("/projects/prjs2156/shared/wsl/vision_datasets/CIFAR100")
EPOCHS = list(range(0, 61, 5))
NUM_CLASSES = 100
COLOR_LIMIT = 0.2
OUT_DIR = Path("recall_prediction/recall_spread")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class_names = datasets.CIFAR100(root=CIFAR100_ROOT, train=False).classes

# recall[epoch] has shape (models, classes)
recall = {epoch: [] for epoch in EPOCHS}
for model_dir in sorted(d for d in ZOO_ROOT.iterdir() if d.is_dir()):
    for line in (model_dir / "result.json").read_text().splitlines():
        line = json.loads(line)
        if line["training_iteration"] in recall:
            recall[line["training_iteration"]].append([line[f"acc_class_{c}"] for c in range(NUM_CLASSES)])

for epoch in EPOCHS:
    epoch_recall = np.array(recall[epoch])
    median_per_class = np.median(epoch_recall, axis=0)
    class_order = np.argsort(-median_per_class)
    sorted_recall = epoch_recall[:, class_order]
    std_per_class = epoch_recall.std(axis=0)
    median_test_accuracy = np.median(epoch_recall.mean(axis=1))
    ranks = np.arange(1, NUM_CLASSES + 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    has_spread = sorted_recall.std(axis=0) > 0  # the violin KDE fails on a constant column
    violins = ax.violinplot(
        sorted_recall[:, has_spread], positions=ranks[has_spread], widths=0.9, showextrema=False
    )
    for body in violins["bodies"]:
        body.set_facecolor("#6baed6")
        body.set_alpha(0.8)
    ax.plot(ranks, median_per_class[class_order], color="#08519c", lw=1.2)
    ax.axhline(median_test_accuracy, color="grey", ls="--", lw=1)
    ax.text(NUM_CLASSES, median_test_accuracy + 0.015, "median test accuracy", color="grey", ha="right")

    for i, rank in enumerate([1, 2, 3]):
        ax.annotate(class_names[class_order[rank - 1]], xy=(rank, median_per_class[class_order[rank - 1]]),
                    xytext=(10 + 15 * i, 0.98 - 0.05 * i), arrowprops=dict(arrowstyle="-", color="grey", lw=0.8))
    for i, rank in enumerate([98, 99, 100]):
        ax.annotate(class_names[class_order[rank - 1]], xy=(rank, median_per_class[class_order[rank - 1]]),
                    xytext=(62 + 12 * i, 0.03), arrowprops=dict(arrowstyle="-", color="grey", lw=0.8))

    ax.set_xlim(0, NUM_CLASSES + 1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel("class, sorted by median recall")
    ax.set_ylabel(f"per-class recall, epoch {epoch}")
    ax.set_title(f"epoch {epoch}: per-class SD {std_per_class.min():.3f}–{std_per_class.max():.3f} "
                 f"over {len(epoch_recall)} models")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"per_class_recall_epoch{epoch:02d}.png", dpi=150)
    plt.close(fig)

    off_diagonal = ~np.eye(NUM_CLASSES, dtype=bool)
    correlation = np.corrcoef(sorted_recall, rowvar=False)
    rng = np.random.default_rng(0)
    shuffled_recall = rng.permuted(sorted_recall, axis=0)  # shuffles each class column on its own
    null_correlation = np.corrcoef(shuffled_recall, rowvar=False)
    r_low, r_high = np.nanpercentile(correlation[off_diagonal], [5, 95])
    null_low, null_high = np.nanpercentile(null_correlation[off_diagonal], [5, 95])

    fig, ax = plt.subplots(figsize=(6, 5))
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("#333333")  # the diagonal
    correlation[~off_diagonal] = np.nan
    image = ax.imshow(correlation, cmap=colormap, vmin=-COLOR_LIMIT, vmax=COLOR_LIMIT,
                      extent=(0.5, NUM_CLASSES + 0.5, NUM_CLASSES + 0.5, 0.5))
    fig.colorbar(image, ax=ax, extend="both", label="Pearson r between models")
    ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_yticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel("class rank")
    ax.set_ylabel("class rank")
    ax.set_title(f"epoch {epoch}: off-diagonal r 5–95%: {r_low:.2f} to {r_high:.2f}\n"
                 f"null {null_low:.2f} to {null_high:.2f}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"per_class_recall_correlation_epoch{epoch:02d}.png", dpi=150)
    plt.close(fig)

    print(f"epoch {epoch}: median test acc {median_test_accuracy:.3f}, "
          f"per-class SD {std_per_class.min():.3f}-{std_per_class.max():.3f}, "
          f"off-diagonal r 5-95% {r_low:.2f} to {r_high:.2f} (min {np.nanmin(correlation):.2f}, "
          f"max {np.nanmax(correlation):.2f}), null {null_low:.2f} to {null_high:.2f}")

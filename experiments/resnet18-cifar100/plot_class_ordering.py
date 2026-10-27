"""Does every model order the 100 CIFAR-100 classes the same way, at every epoch?

Four panels, over epochs 5..60 of the full zoo (epoch 0 is chance level, so its
order has no meaning):
  a) consensus rank of each class (by mean recall over the models) per epoch, one
     line per class, coloured by its rank at epoch 60. Flat lines = a stable order.
  b) per epoch, the Spearman correlation of each model's class order with the
     consensus order of that epoch. Grey: the same correlation for a simulated model
     that has the consensus order and only test-image noise (each class recall drawn
     as Binomial(100, consensus mean recall) / 100). A model below the grey band
     orders the classes differently by more than the test-image noise.
  c, d) the class rank of every model at epoch 20 and 60: rows are models sorted by
     their correlation from b), columns are classes in consensus order. A model with
     another order shows up as a row that breaks the left-to-right colour gradient.

Reads acc_class_0..99 from each model's result.json and writes
recall_prediction/recall_spread/class_ordering.png.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ZOO_ROOT = Path(
    "/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform"
)
EPOCHS = list(range(5, 61, 5))
HEATMAP_EPOCHS = [20, 60]
NUM_CLASSES = 100
OUT_DIR = Path("recall_prediction/recall_spread")
OUT_DIR.mkdir(parents=True, exist_ok=True)

recall = {epoch: [] for epoch in EPOCHS}  # recall[epoch] has shape (models, classes)
for model_dir in sorted(d for d in ZOO_ROOT.iterdir() if d.is_dir()):
    for line in (model_dir / "result.json").read_text().splitlines():
        line = json.loads(line)
        if line["training_iteration"] in recall:
            recall[line["training_iteration"]].append([line[f"acc_class_{c}"] for c in range(NUM_CLASSES)])
recall = {epoch: np.array(values) for epoch, values in recall.items()}


def descending_ranks(values):
    """Rank 1 = highest recall; ties get their average rank."""
    return rankdata(-values, axis=-1)


def spearman_with(ranks, reference_ranks):
    """Spearman correlation of each row of ranks with reference_ranks."""
    ranks = ranks - ranks.mean(axis=-1, keepdims=True)
    reference_ranks = reference_ranks - reference_ranks.mean()
    return ranks @ reference_ranks / (np.linalg.norm(ranks, axis=-1) * np.linalg.norm(reference_ranks))


consensus_ranks = {epoch: descending_ranks(recall[epoch].mean(axis=0)) for epoch in EPOCHS}
model_ranks = {epoch: descending_ranks(recall[epoch]) for epoch in EPOCHS}
model_correlation = {epoch: spearman_with(model_ranks[epoch], consensus_ranks[epoch]) for epoch in EPOCHS}
random_generator = np.random.default_rng(0)
noise_only_correlation = {
    epoch: spearman_with(
        descending_ranks(random_generator.binomial(100, recall[epoch].mean(axis=0), size=(1000, NUM_CLASSES)) / 100),
        consensus_ranks[epoch],
    )
    for epoch in EPOCHS
}

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
colormap = plt.get_cmap("viridis_r")

# a) consensus rank per epoch
ax = axes[0, 0]
for class_index in range(NUM_CLASSES):
    final_rank = consensus_ranks[60][class_index]
    ax.plot(EPOCHS, [consensus_ranks[e][class_index] for e in EPOCHS],
            color=colormap((final_rank - 1) / (NUM_CLASSES - 1)), lw=0.8, alpha=0.8)
ax.invert_yaxis()
ax.set_xticks(EPOCHS)
ax.set_xlabel("epoch")
ax.set_ylabel("consensus rank of the class (1 = highest mean recall)")
ax.set_title(f"a) consensus class order; Spearman(epoch 5, epoch 60) = "
             f"{spearman_with(consensus_ranks[5], consensus_ranks[60]):.2f}")

# b) per-model agreement with the consensus of the same epoch
ax = axes[0, 1]
for i, epoch in enumerate(EPOCHS):
    low, high = np.percentile(noise_only_correlation[epoch], [1, 99])
    ax.fill_between([i + 0.6, i + 1.4], low, high, color="grey", alpha=0.35, lw=0)
violins = ax.violinplot([model_correlation[e] for e in EPOCHS], showextrema=False, widths=0.8)
for body in violins["bodies"]:
    body.set_facecolor("#6baed6")
    body.set_alpha(0.8)
ax.scatter(range(1, len(EPOCHS) + 1), [model_correlation[e].min() for e in EPOCHS], color="#08519c", s=12,
           label="least typical model")
ax.set_xticks(range(1, len(EPOCHS) + 1), EPOCHS)
ax.set_xlabel("epoch")
ax.set_ylabel("Spearman(model class order, consensus order)")
ax.set_title("b) agreement of each model with the consensus\n(grey: consensus order + test-image noise, 1st–99th pct)")
ax.legend(frameon=False, loc="lower right")

# c, d) the class rank of every model
for ax, epoch in zip(axes[1], HEATMAP_EPOCHS):
    model_order = np.argsort(-model_correlation[epoch])
    class_order = np.argsort(consensus_ranks[epoch])
    image = ax.imshow(model_ranks[epoch][model_order][:, class_order], aspect="auto", cmap=colormap,
                      interpolation="nearest", vmin=1, vmax=NUM_CLASSES)
    ax.set_xlabel(f"class, in consensus order of epoch {epoch}")
    ax.set_ylabel("model, sorted by agreement (top = most typical)")
    ax.set_title(f"{'c' if epoch == HEATMAP_EPOCHS[0] else 'd'}) class rank of each model, epoch {epoch}")
fig.colorbar(image, ax=axes[1], label="class rank in the model (1 = highest recall)", shrink=0.8)

fig.savefig(OUT_DIR / "class_ordering.png", dpi=150, bbox_inches="tight")

for epoch in EPOCHS:
    noise_floor = np.percentile(noise_only_correlation[epoch], 1)
    print(f"epoch {epoch:2d}: Spearman with consensus median {np.median(model_correlation[epoch]):.3f}, "
          f"min {model_correlation[epoch].min():.3f}; noise-only 1st pct {noise_floor:.3f}; "
          f"models below it {np.mean(model_correlation[epoch] < noise_floor):.1%}; "
          f"consensus vs epoch 60 {spearman_with(consensus_ranks[epoch], consensus_ranks[60]):.3f}")

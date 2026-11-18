"""Paper figure: class rank of every model at an early, the edited and the final epoch.

One panel per epoch in EPOCHS (5, 20, 60). Rows are the 1000 models, sorted by the
Spearman correlation of their class order with the consensus order (top = most
typical); columns are the 100 classes in consensus order (by mean recall). The colour
is the rank of the class within the model (1 = highest recall). Ranks remove each
model's overall accuracy, so the panels show the model x class interaction: a smooth
left-to-right gradient means that all models order the classes the same way.

Also prints, for every 5th epoch, the statistics of the appendix table: test accuracy
(mean, sd, min, max), train accuracy, and the share of the recall variance that is
model x class interaction (sum of squares of the double-centred recalls over the total
sum of squares around the grand mean).

Reads acc_class_0..99 from each model's result.json (written by
evaluate_per_class_recall_cifar100_resnet18.py) and writes
recall_prediction/recall_spread/class_rank_epochs.pdf.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

ZOO_ROOT = Path(
    "/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform"
)
EPOCHS = [5, 20, 60]
TABLE_EPOCHS = list(range(0, 61, 5))
NUM_CLASSES = 100
OUT_DIR = Path("recall_prediction/recall_spread")
OUT_DIR.mkdir(parents=True, exist_ok=True)

recall = {epoch: [] for epoch in TABLE_EPOCHS}  # recall[epoch] has shape (models, classes)
test_accuracy = {epoch: [] for epoch in TABLE_EPOCHS}
train_accuracy = {epoch: [] for epoch in TABLE_EPOCHS}
for model_dir in sorted(d for d in ZOO_ROOT.iterdir() if d.is_dir()):
    for line in (model_dir / "result.json").read_text().splitlines():
        line = json.loads(line)
        epoch = line["training_iteration"]
        if epoch in recall:
            recall[epoch].append([line[f"acc_class_{c}"] for c in range(NUM_CLASSES)])
            test_accuracy[epoch].append(line["test_acc"])
            train_accuracy[epoch].append(line["train_acc"])
recall = {epoch: np.array(values) for epoch, values in recall.items()}


def descending_ranks(values):
    """Rank 1 = highest recall; ties get their average rank."""
    return rankdata(-values, axis=-1)


def spearman_with(ranks, reference_ranks):
    """Spearman correlation of each row of ranks with reference_ranks."""
    ranks = ranks - ranks.mean(axis=-1, keepdims=True)
    reference_ranks = reference_ranks - reference_ranks.mean()
    return ranks @ reference_ranks / (np.linalg.norm(ranks, axis=-1) * np.linalg.norm(reference_ranks))


def interaction_share(recall_matrix):
    """Share of the recall variance that is neither a model nor a class effect."""
    double_centred = (recall_matrix - recall_matrix.mean(axis=1, keepdims=True)
                      - recall_matrix.mean(axis=0, keepdims=True) + recall_matrix.mean())
    return (double_centred ** 2).sum() / ((recall_matrix - recall_matrix.mean()) ** 2).sum()


plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8})
fig, axes = plt.subplots(1, len(EPOCHS), figsize=(6.5, 2.4), sharey=True, constrained_layout=True)
colormap = plt.get_cmap("viridis_r")
for ax, epoch in zip(axes, EPOCHS):
    model_ranks = descending_ranks(recall[epoch])
    consensus_ranks = descending_ranks(recall[epoch].mean(axis=0))
    model_order = np.argsort(-spearman_with(model_ranks, consensus_ranks))
    class_order = np.argsort(consensus_ranks)
    image = ax.imshow(model_ranks[model_order][:, class_order], aspect="auto", cmap=colormap,
                      interpolation="nearest", vmin=1, vmax=NUM_CLASSES)
    ax.set_title(f"epoch {epoch} (interaction {interaction_share(recall[epoch]):.0%})")
    ax.set_xlabel("class (consensus order)")
    ax.set_xticks([0, 49, 99], [1, 50, 100])
axes[0].set_ylabel("model (by agreement)")
axes[0].set_yticks([0, 499, 999], [1, 500, 1000])
fig.colorbar(image, ax=axes, label="class rank in model", ticks=[1, 50, 100], shrink=0.9)
fig.savefig(OUT_DIR / "class_rank_epochs.pdf", dpi=300)
print(f"wrote {OUT_DIR / 'class_rank_epochs.pdf'}")

print("epoch & test acc mean & sd & min & max & train acc & interaction share")
for epoch in TABLE_EPOCHS:
    accuracy = np.array(test_accuracy[epoch])
    print(f"{epoch} & {accuracy.mean():.3f} & {accuracy.std():.3f} & {accuracy.min():.3f} & {accuracy.max():.3f}"
          f" & {np.mean(train_accuracy[epoch]):.3f} & {interaction_share(recall[epoch]):.2f}")

"""Distribution of per-class recall over the models of the four unthi zoos, train and test.

Question: is there enough spread in recall per class over the models, or can a predictor
reach a low loss by predicting the class mean? One panel per zoo (columns) and checkpoint
(rows: 4 and 8, the training data of the recall transformer). Each class is a split violin:
left half train, right half test (split.json of the zoo), with the median as a line.
Models with recall -999 (no recall) are left out.
Prints the std of recall over the train models per class.
Writes recall_distribution.png. Run from this directory (reads about 120k result.json files,
a few minutes).
"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZOOS = ["mnist", "fmnist", "cifar10", "svhn"]
# as in build_tokens.py
ZOO_ROOTS = {zoo: Path(f"/scratch-shared/mmiddelkoop/unthi_zoo/unthi_{zoo}") for zoo in ZOOS}
ZOO_ROOTS["fmnist"] = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_fmnist")
CHECKPOINTS = [4, 8]
SPLITS = {"train": "#2a78d6", "test": "#eb6834"}
N_CLASSES = 10
SENTINEL = -999


def recall_at_checkpoints(model_dir):
    """[len(CHECKPOINTS), N_CLASSES] recall of one model"""
    lines = [json.loads(line) for line in (model_dir / "result.json").open()]
    line_of_iteration = {line["training_iteration"]: line for line in lines}
    return [[line_of_iteration[c][f"acc_class_{k}"] for k in range(N_CLASSES)] for c in CHECKPOINTS]


recall = {}  # (zoo, split) -> [models, checkpoints, classes]
for zoo in ZOOS:
    split = json.load((ZOO_ROOTS[zoo] / "split.json").open())
    for split_name in SPLITS:
        with ThreadPoolExecutor(32) as pool:
            values = np.array(list(pool.map(recall_at_checkpoints, [ZOO_ROOTS[zoo] / d for d in split[split_name]])))
        recall[zoo, split_name] = values[(values != SENTINEL).all(axis=(1, 2))]
        print(f"{zoo} {split_name}: {len(recall[zoo, split_name])} models")

fig, axes = plt.subplots(len(CHECKPOINTS), len(ZOOS), figsize=(18, 7), sharey=True)
for column, zoo in enumerate(ZOOS):
    for row, checkpoint in enumerate(CHECKPOINTS):
        ax = axes[row, column]
        for side, (split_name, color) in enumerate(SPLITS.items()):
            values = recall[zoo, split_name][:, row]  # [models, classes]
            parts = ax.violinplot(values, positions=range(N_CLASSES), widths=0.9, showextrema=False)
            for class_index, body in enumerate(parts["bodies"]):
                vertices = body.get_paths()[0].vertices
                # keep the left half for train, the right half for test
                vertices[:, 0] = np.clip(vertices[:, 0], *((-np.inf, class_index) if side == 0 else (class_index, np.inf)))
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.6)
            medians = np.median(values, axis=0)
            offsets = (-0.4, 0) if side == 0 else (0, 0.4)
            for class_index, median in enumerate(medians):
                ax.plot([class_index + offsets[0], class_index + offsets[1]], [median, median], color="black", linewidth=1)
        ax.set_title(f"{zoo}, checkpoint {checkpoint}")
        ax.set_xticks(range(N_CLASSES))
        ax.grid(axis="y", alpha=0.3)
        if row == len(CHECKPOINTS) - 1:
            ax.set_xlabel("class")
        if column == 0:
            ax.set_ylabel("recall")
        print(f"{zoo} ep{checkpoint} train std per class:", np.round(recall[zoo, "train"][:, row].std(axis=0), 3))
handles = [matplotlib.patches.Patch(color=color, alpha=0.6, label=name) for name, color in SPLITS.items()]
handles.append(matplotlib.lines.Line2D([], [], color="black", linewidth=1, label="median"))
fig.legend(handles=handles, loc="upper right")
fig.suptitle("Per-class recall over the models of each unthi zoo")
fig.tight_layout(rect=(0, 0, 0.95, 1))
fig.savefig("recall_distribution.png", dpi=150)
print("Saved recall_distribution.png")

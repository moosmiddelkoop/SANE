"""Build <zoo>_labelled.npz and edit_sets_<zoo>.npz from a Small-CNN-Zoo folder with per-class labels.

usage: python build_zoo.py <zoo> [root] [acc_floor_unlearn] [recall_floor_unlearn]   (SETS_ONLY=1 reuses an existing <zoo>_labelled.npz)
  zoo in mnist | fashion_mnist | cifar10 | svhn_cropped

<zoo>_labelled.npz keys (same layout as mnist_zoo_labelled.npz):
  theta (3, M, 4970) float32   checkpoints epoch 0 / 20 / 86
  recall (3, M, 10), acc (3, M), act (M,) int 0=relu 1=tanh
  split (M,) int 0=train 1=val 2=test  (50/25/25, permutation seed 0 over model order)
  modeldir (M,) str, row86 (M,) int (row in weights.npy), configs: cfg_<name> arrays
  ckpt_steps = [0, 20, 86]
edit_sets_<zoo>.npz keys:
  unlearn_<split>_c<k>, uplearn_<split>_c<k>: model indices (into M), up to 200 each, rng seed 0
  plus eligible counts in 'counts' (json string)
"""
import sys, os, json
import numpy as np, pandas as pd

zoo = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "/path/to/small_cnn_zoo"
acc_floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
rec_floor = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
SETS_ONLY = os.environ.get("SETS_ONLY") == "1"
CC = [f"accuracy_class_{k}" for k in range(10)]
CFG = ["activation", "b_init", "dropout", "init_std", "l2reg", "learning_rate", "optimizer", "train_fraction", "w_init"]

met = pd.read_csv(f"{root}/{zoo}/metrics.csv", usecols=["modeldir", "step", "test_accuracy"] + [f"config.{c}" for c in CFG])
met["row"] = np.arange(len(met))
labs = {}
for step, f in [(0, "early"), (20, "middle"), (86, "final")]:
    x = pd.read_csv(f"{root}/{zoo}/metrics_merged_{f}.csv", usecols=["modeldir", "step", "test_accuracy"] + CC)
    x = x[x[CC[0]].notna()]
    assert (x.step == step).all(), (zoo, f, x.step.unique())
    labs[step] = x.set_index("modeldir")
common = labs[0].index.intersection(labs[20].index).intersection(labs[86].index)
base = met[met.step == 86].set_index("modeldir").loc[common]
rows = {s: met[met.step == s].set_index("modeldir").loc[common, "row"].values.astype(int) for s in (0, 20, 86)}
M = len(common)
print(zoo, "models with all three labelled checkpoints:", M)

if SETS_ONLY:
    z = np.load(f"{zoo}_labelled.npz"); recall, acc, split = z["recall"], z["acc"], z["split"]
else:
  W = np.load(f"{root}/{zoo}/weights.npy", mmap_mode="r")
  theta = np.empty((3, M, 4970), dtype=np.float32)
  recall = np.empty((3, M, 10), dtype=np.float32)
  acc = np.empty((3, M), dtype=np.float32)
  for i, s in enumerate((0, 20, 86)):
      order = np.argsort(rows[s])
      theta[i, order] = W[rows[s][order]]
      recall[i] = labs[s].loc[common, CC].values
      acc[i] = labs[s].loc[common, "test_accuracy"].values
  act = (base["config.activation"].values == "tanh").astype(np.int64)
  perm = np.random.default_rng(0).permutation(M)
  split = np.empty(M, dtype=np.int64)
  split[perm[: M // 2]] = 0
  split[perm[M // 2: 3 * M // 4]] = 1
  split[perm[3 * M // 4:]] = 2
  out = dict(theta=theta, recall=recall, acc=acc, act=act, split=split, modeldir=np.asarray(common, dtype=str),
             row86=rows[86], ckpt_steps=np.array([0, 20, 86]))
  for c in CFG:
      v = base[f"config.{c}"].values
      out[f"cfg_{c}"] = np.asarray(v).astype(str) if np.asarray(v).dtype.kind == "O" else np.asarray(v)
  np.savez(f"{zoo}_labelled.npz", **out)

# edit sets at epoch 86
r86, a86 = recall[2], acc[2]
rng = np.random.default_rng(0)
sets, counts = {}, {}
for sp_name, sp in [("train", 0), ("val", 1), ("test", 2)]:
    for k in range(10):
        un = np.where((split == sp) & (a86 >= acc_floor) & (r86[:, k] >= rec_floor))[0]
        up = np.where((split == sp) & (r86[:, k] >= 0.1) & (r86[:, k] <= 0.8) & (r86.min(1) > 0))[0]
        counts[f"unlearn_{sp_name}_c{k}"] = int(len(un)); counts[f"uplearn_{sp_name}_c{k}"] = int(len(up))
        sets[f"unlearn_{sp_name}_c{k}"] = rng.choice(un, min(200, len(un)), replace=False) if len(un) else un
        sets[f"uplearn_{sp_name}_c{k}"] = rng.choice(up, min(200, len(up)), replace=False) if len(up) else up
np.savez(f"edit_sets_{zoo}.npz", counts=json.dumps(dict(counts, acc_floor=acc_floor, rec_floor=rec_floor)), **sets)
print("acc>=0.5 at ep86: %.3f  acc<0.2: %.3f  median acc %.3f" % ((a86 >= 0.5).mean(), (a86 < 0.2).mean(), np.median(a86)))
print("test unlearn-eligible per class:", [counts[f"unlearn_test_c{k}"] for k in range(10)])
print("test uplearn-eligible per class:", [counts[f"uplearn_test_c{k}"] for k in range(10)])

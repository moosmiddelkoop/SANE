"""Token cache for the end-to-end recall transformer, one unthi zoo (mnist, fmnist, cifar10 or svhn).

Loads every model of the zoo with DatasetTokens, with the settings of the SANE preprocessing
(data/preprocess_dataset_smallcnnzoo.py): canonical form, per-layer standardization, no
batch norm, tokensize 145, weight threshold 100. The standardization statistics come from all
9 checkpoints (0-8) of a split, as in SANE pretraining. Only checkpoints 4 and 8 are kept:
per-class recall is valid at 0, 4 and 8, and 0 is the random init. A checkpoint that the
weight threshold drops is not in the cache. A checkpoint without per-class recall (-999, e.g.
NN_unthi_fashion_mnist_11172982_9999 at all iterations) is not in the cache either; the log
names these models.

Zoo: Fashion-MNIST is read from /projects (extracted by silic); the other zoos from their
scratch copy, which build_tokens.sh unzips. Each zoo needs <zoo>/split.json.
Writes to /projects/prjs2156/shared/wsl/unthi_zoo/unthi_<zoo>_preprocessed/:
  tokens_ep4-8.pt: tokens [N, n_tokens, 145], positions [n_tokens, 3], Y [N, 10] per-class
    recall, epoch [N], model_dir [N], split [N] ("train"/"val"/"test"), split_id
  zoo_split.json: a copy of <zoo>/split.json, as scratch deletes files after two weeks
Run: sbatch build_tokens.sh <zoo> (CPU, about 1 h, plus about 1 h to unzip).
"""

import os

NUM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import gc
import logging
import shutil
from argparse import ArgumentParser
from pathlib import Path

import torch

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--zoo", choices=["mnist", "fmnist", "cifar10", "svhn"], required=True)
ZOO = parser.parse_args().zoo

PROJECT_DIR = Path("/projects/prjs2156/shared/wsl/unthi_zoo")
ZOO_ROOT = PROJECT_DIR / "unthi_fmnist" if ZOO == "fmnist" else Path(f"/scratch-shared/mmiddelkoop/unthi_zoo/unthi_{ZOO}")
OUT_DIR = PROJECT_DIR / f"unthi_{ZOO}_preprocessed"
CACHE_PT = OUT_DIR / "tokens_ep4-8.pt"
LOADED_EPOCHS = list(range(9))  # standardization statistics as in SANE pretraining
KEPT_EPOCHS = [4, 8]
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]
SENTINEL = -999
SEED = 0

seed_everything(SEED)
tokens, Y, epoch, model_dir, split_of_sample, models_without_recall = [], [], [], [], [], set()
for split in ["train", "val", "test"]:
    logging.info(f"Building DatasetTokens split={split}")
    ds = DatasetTokens(
        root=ZOO_ROOT,
        epoch_lst=LOADED_EPOCHS,
        mode="vector",
        permutation_spec=smallcnnzoo_permutation_spec(),
        map_to_canonical=True,
        standardize=True,
        tokensize=145,
        train_val_test=split,
        weight_threshold=100,
        max_samples=None,
        property_keys={"result_keys": ACC_CLASS_KEYS, "config_keys": []},
        num_threads=NUM_THREADS,
        verbosity=3,
        getitem="tokens+props",
        ignore_bn=True,
    )
    for model_index, model_epochs in enumerate(ds.epochs):
        for epoch_index, model_epoch in enumerate(model_epochs):
            if model_epoch not in KEPT_EPOCHS:
                continue
            recall = [ds.properties[key][model_index][epoch_index] for key in ACC_CLASS_KEYS]
            if SENTINEL in recall:
                models_without_recall.add(Path(ds.paths[model_index][0]).name)
                continue
            tokens.append(ds.data[model_index][epoch_index])
            Y.append(recall)
            epoch.append(model_epoch)
            model_dir.append(Path(ds.paths[model_index][0]).name)
            split_of_sample.append(split)
    positions, split_id = ds.positions, ds.split_id
    del ds
    gc.collect()

logging.info(f"Dropped models without per-class recall ({len(models_without_recall)}): {sorted(models_without_recall)}")
Y = torch.tensor(Y)
torch.save(
    {
        "tokens": torch.stack(tokens),
        "positions": positions,
        "Y": Y,
        "epoch": torch.tensor(epoch),
        "model_dir": model_dir,
        "split": split_of_sample,
        "split_id": split_id,
    },
    CACHE_PT,
)
shutil.copy(ZOO_ROOT / "split.json", OUT_DIR / "zoo_split.json")
logging.info(f"Saved {len(Y)} samples to {CACHE_PT}")

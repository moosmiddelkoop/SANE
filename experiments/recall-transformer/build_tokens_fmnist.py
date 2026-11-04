"""Token cache for the end-to-end recall transformer, Fashion-MNIST unthi zoo.

Loads every model of the zoo with DatasetTokens, with the settings of the SANE preprocessing
(data/preprocess_dataset_smallcnnzoo.py): canonical form, per-layer standardization, no
batch norm, tokensize 145, weight threshold 100. The standardization statistics come from all
9 checkpoints (0-8) of a split, as in SANE pretraining. Only checkpoints 4 and 8 are kept:
per-class recall is valid at 0, 4 and 8, and 0 is the random init. A checkpoint that the
weight threshold drops is not in the cache.

Writes cache/tokens_fmnist_ep4-8.pt:
  tokens [N, n_tokens, 145], positions [n_tokens, 3], Y [N, 10] per-class recall,
  epoch [N], model_dir [N], split [N] ("train"/"val"/"test"), split_id
Needs <zoo>/split.json. Run: sbatch build_tokens_fmnist.sh (CPU, about 1-2 h).
"""

import os

NUM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import gc
import logging
from pathlib import Path

import torch

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_fmnist")
CACHE_PT = Path("cache/tokens_fmnist_ep4-8.pt")
LOADED_EPOCHS = list(range(9))  # standardization statistics as in SANE pretraining
KEPT_EPOCHS = [4, 8]
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(10)]
SENTINEL = -999
SEED = 0

seed_everything(SEED)
tokens, Y, epoch, model_dir, split_of_sample = [], [], [], [], []
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
            if model_epoch in KEPT_EPOCHS:
                tokens.append(ds.data[model_index][epoch_index])
                Y.append([ds.properties[key][model_index][epoch_index] for key in ACC_CLASS_KEYS])
                epoch.append(model_epoch)
                model_dir.append(Path(ds.paths[model_index][0]).name)
                split_of_sample.append(split)
    positions, split_id = ds.positions, ds.split_id
    del ds
    gc.collect()

Y = torch.tensor(Y)
assert (Y != SENTINEL).all(), "per-class recall is missing at a kept checkpoint"
CACHE_PT.parent.mkdir(exist_ok=True)
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
logging.info(f"Saved {len(Y)} samples to {CACHE_PT}")

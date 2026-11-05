"""Token cache for the end-to-end recall transformer, ResNet-18 CIFAR-100 zoo (1000 models).

Loads every model with DatasetTokens, with the settings of the all-epoch SANE preprocessing
(data/preprocess_dataset_resnet18_cifar100_tk288_allepochs.py): canonical form, per-layer
standardization, no batch norm, tokensize 288, no weight threshold, and the same 28 model chunks
per split. The standardization statistics come from all 13 checkpoints (0, 5, ..., 60) of a
chunk, as in that preprocessing. Only checkpoints 5, 10, ..., 60 are kept (0 is the random
init). A checkpoint without per-class recall (-999) is not in the cache; the log names it.

One checkpoint is 39304 tokens x 288 (fp16, 22.6 MB), so the 12000 checkpoints (~272 GB) do not
fit in RAM. The tokens are appended chunk by chunk to one raw fp16 file, which the training
script reads as a memory map. Writes to OUT_DIR (scratch: deleted after two weeks unused):
  tokens_ep5-60.bin: tokens [N, n_tokens, 288], fp16, row-major
  tokens_ep5-60_meta.pt: shape, positions [n_tokens, 3], Y [N, 100] per-class recall,
    epoch [N], model_dir [N], split [N] ("train"/"val"/"test"), split_id
Run: sbatch build_tokens_resnet18.sh (CPU, about 1.5-2 h).
"""

import os

NUM_THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.git_re_basin.git_re_basin import resnet18_permutation_spec
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--out_dir", default="/scratch-shared/mmiddelkoop/recall_transformer/resnet18_cifar100")
parser.add_argument("--max_chunks", type=int, default=None, help="smoke test: only the first chunks of each split")
args = parser.parse_args()

ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform")
OUT_DIR = Path(args.out_dir)
TOKENS_BIN = OUT_DIR / "tokens_ep5-60.bin"
META_PT = OUT_DIR / "tokens_ep5-60_meta.pt"
LOADED_EPOCHS = list(range(0, 61, 5))  # standardization statistics as in the SANE preprocessing
KEPT_EPOCHS = list(range(5, 61, 5))
N_CHUNKS = 28  # as in the SANE preprocessing
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(100)]
SENTINEL = -999
SEED = 0

seed_everything(SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)
Y, epoch, model_dir, split_of_sample, models_without_recall = [], [], [], [], set()
with TOKENS_BIN.open("wb") as tokens_file:
    for split in ["train", "val", "test"]:
        for chunk_index in range(args.max_chunks or N_CHUNKS):
            logging.info(f"Building DatasetTokens split={split}, chunk {chunk_index + 1} of {N_CHUNKS}")
            ds = DatasetTokens(
                root=ZOO_ROOT,
                epoch_lst=LOADED_EPOCHS,
                mode="vector",
                permutation_spec=resnet18_permutation_spec(),
                map_to_canonical=True,
                standardize=True,
                tokensize=288,
                train_val_test=split,
                weight_threshold=float("inf"),
                max_samples=None,
                model_chunk=(chunk_index, N_CHUNKS),
                property_keys={"result_keys": ACC_CLASS_KEYS, "config_keys": []},
                num_threads=NUM_THREADS,
                verbosity=3,
                precision="16",
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
                    tokens_file.write(ds.data[model_index][epoch_index].numpy().tobytes())
                    Y.append(recall)
                    epoch.append(model_epoch)
                    model_dir.append(Path(ds.paths[model_index][0]).name)
                    split_of_sample.append(split)
            positions, split_id = ds.positions, ds.split_id
            n_tokens, tokensize = ds.data[0][0].shape
            del ds
            gc.collect()

logging.info(f"Dropped models without per-class recall ({len(models_without_recall)}): {sorted(models_without_recall)}")
shape = (len(Y), n_tokens, tokensize)
assert TOKENS_BIN.stat().st_size == 2 * shape[0] * shape[1] * shape[2], "token file size does not match the samples"
torch.save(
    {
        "shape": shape,
        "positions": positions,
        "Y": torch.tensor(Y),
        "epoch": torch.tensor(epoch),
        "model_dir": model_dir,
        "split": split_of_sample,
        "split_id": split_id,
    },
    META_PT,
)
logging.info(f"Saved {len(Y)} samples of shape {shape[1:]} to {TOKENS_BIN}")

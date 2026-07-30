"""One-time consolidation of a preprocessed zoo: packs the per-sample .pt files
in dataset_torch.{train,val,test} into stacked tensors and saves them as
<src>/consolidated/dataset.pt (a {"trainset","valset","testset"} dict of
TensorSamplingDataset). Training then loads the whole dataset with one
sequential read instead of 100k+ small-file reads per epoch.

Usage: python consolidate_preprocessed.py /path/to/preprocessed_zoo
"""

import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from SANE.datasets.dataset_sampling_preprocessed import TensorSamplingDataset

WORKERS = 32


def consolidate_split(split_dir):
    with os.scandir(split_dir) as it:
        files = sorted(e.path for e in it if e.is_file())
    n = len(files)
    first = torch.load(files[0])
    # preallocate and fill in place: no 2x peak memory from torch.stack
    out = {
        k: torch.empty((n, *v.shape), dtype=v.dtype) for k, v in first.items()
    }

    def fill(i):
        item = torch.load(files[i])
        for k in out:
            out[k][i] = item[k]

    with ThreadPoolExecutor(WORKERS) as ex:
        for i, _ in enumerate(ex.map(fill, range(n))):
            if i % 10000 == 0:
                print(f"{split_dir.name}: {i}/{n}", flush=True)
    print(f"{split_dir.name}: done ({n} samples)", flush=True)
    return out


def main():
    src = Path(sys.argv[1])
    dst = src.joinpath("consolidated")
    dst.mkdir(exist_ok=True)

    datasets = {}
    for split, name in (("train", "trainset"), ("val", "valset"), ("test", "testset")):
        t = consolidate_split(src.joinpath(f"dataset_torch.{split}"))
        datasets[name] = TensorSamplingDataset(t["w"], t["m"], t["p"], t["props"])

    print("saving dataset.pt ...", flush=True)
    torch.save(datasets, dst.joinpath("dataset.pt"))

    # keep the info/normalization jsons next to the new dataset.pt so path
    # rewrites like dataset.pt -> dataset_info_test.json keep working
    for f in src.glob("dataset_*.json"):
        shutil.copy(f, dst)
    print(f"done: {dst}", flush=True)


if __name__ == "__main__":
    main()

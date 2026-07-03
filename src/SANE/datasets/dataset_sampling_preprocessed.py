import os
import torch
from pathlib import Path
from torch.utils.data import Dataset


class PreprocessedSamplingDataset(Dataset):
    def __init__(self, zoo_paths, split="train", transforms=None):
        self.zoo_paths = zoo_paths
        self.split = split
        self.datasets = self.load_datasets(zoo_paths)
        self.transforms = transforms
        self.samples = self.collect_samples(self.datasets)

    def load_datasets(self, zoo_paths):
        datasets = []
        for path in zoo_paths:
            directory_path = (
                Path(path).joinpath(f"dataset_torch.{self.split}").absolute()
            )
            if os.path.isdir(directory_path):
                datasets.append(directory_path)
            else:
                raise NotADirectoryError(f"Directory not found: {directory_path}")
        return datasets

    def collect_samples(self, datasets):
        samples = []
        for dataset in datasets:
            # os.scandir returns dirent type info from the single directory read,
            # so entry.is_file() avoids a separate stat() syscall per file. On GPFS
            # this turns an O(n) metadata-bound stat-walk (slow for 100k+ samples)
            # into a single cheap directory scan.
            with os.scandir(dataset) as it:
                for entry in it:
                    if entry.is_file():
                        samples.append(entry.path)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        item = torch.load(file_path)

        ddx = item["w"]
        mask = item["m"]
        pos = item["p"]
        props = item["props"]

        if self.transforms:
            ddx, mask, pos = self.transforms(ddx, mask, pos)
        return ddx, mask, pos, props


class TensorSamplingDataset(Dataset):
    """All samples stacked into in-RAM tensors (built by data/consolidate_preprocessed.py).

    Avoids the per-sample torch.load of PreprocessedSamplingDataset, which on
    GPFS costs one metadata-bound small-file read per sample per epoch.
    """

    def __init__(self, w, m, p, props, transforms=None):
        self.w = w
        self.m = m
        self.p = p
        self.props = props
        self.transforms = transforms

    def __len__(self):
        return self.w.shape[0]

    def __getitem__(self, idx):
        ddx, mask, pos = self.w[idx], self.m[idx], self.p[idx]
        if self.transforms:
            ddx, mask, pos = self.transforms(ddx, mask, pos)
        return ddx, mask, pos, self.props[idx]

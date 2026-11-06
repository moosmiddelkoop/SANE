"""Writes tokens_ep5-60_fc.bin: the fc tokens (the last 200 of each sample) of the token cache.

train_class_token_transformer_resnet18.py --input fc reads this 1.4 GB file. Reading the fc part
from tokens_ep5-60.bin itself is a strided read of the 272 GB file: about 80 min with one reader,
so this script reads it once with 16 readers. Same sample order as tokens_ep5-60_meta.pt.
Run: sbatch extract_fc_tokens_resnet18.sh
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path("/scratch-shared/mmiddelkoop/recall_transformer/resnet18_cifar100")
SAMPLES_PER_READ = 50

meta = torch.load(CACHE_DIR / "tokens_ep5-60_meta.pt")
n_samples, n_tokens, tokensize = meta["shape"]
n_body_tokens = int((meta["positions"][:, 1] < meta["positions"][:, 1].max()).sum())  # fc is the last layer
cache = np.memmap(CACHE_DIR / "tokens_ep5-60.bin", dtype=np.float16, mode="r", shape=(n_samples, n_tokens, tokensize))

with ThreadPoolExecutor(16) as pool:
    parts = list(pool.map(lambda start: np.array(cache[start : start + SAMPLES_PER_READ, n_body_tokens:]), range(0, n_samples, SAMPLES_PER_READ)))
np.concatenate(parts).tofile(CACHE_DIR / "tokens_ep5-60_fc.bin")
print(f"wrote {n_samples} x {n_tokens - n_body_tokens} x {tokensize}")

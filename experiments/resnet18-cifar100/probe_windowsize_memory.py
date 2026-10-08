# Measures peak GPU memory and step time of SANE pretraining train steps for one
# (windowsize, batchsize) pair, on synthetic tokens (no dataset.pt needed).
# Model config = pretrain_sane_cifar100_resnet18.py. One pair per process, so an OOM
# does not leave memory behind for the next pair (see the sibling .sh).
import time
from argparse import ArgumentParser

import torch

from SANE.models.def_AE_module import AEModule

N_STEPS = 10

parser = ArgumentParser()
parser.add_argument("--windowsize", type=int, required=True)
parser.add_argument("--batchsize", type=int, required=True)
args = parser.parse_args()

config = {
    "seed": 32,
    "device": "cuda",
    "training::precision": "amp",
    "trainset::batchsize": args.batchsize,
    "training::windowsize": args.windowsize,
    "ae:transformer_type": "gpt2",
    "model::compile": True,
    "ae:i_dim": 288,
    "ae:lat_dim": 128,
    "ae:max_positions": [55000, 100, 550],
    "ae:d_model": 2048,
    "ae:nhead": 16,
    "ae:num_layers": 8,
    "optim::optimizer": "adamw",
    "optim::lr": 1e-4,
    "optim::wd": 3e-9,
    "training::temperature": 0.1,
    "training::gamma": 0.05,
    "training::reduction": "mean",
    "training::contrast": "simclr",
    "training::gradient_clipping": "norm",
    "training::gradient_clipp_value": 2.0,
}
module = AEModule(config)
module.model.train()


def random_view():
    x = torch.randn(args.batchsize, args.windowsize, 288, device="cuda")
    m = torch.ones_like(x, dtype=torch.bool)
    p = torch.stack([torch.randint(0, n, (args.batchsize, args.windowsize), device="cuda") for n in config["ae:max_positions"]], dim=-1)
    return x, m, p


try:
    for step in range(N_STEPS):
        if step == 2:  # skip compile / warmup steps in the timing
            torch.cuda.synchronize()
            start = time.time()
        module.train_step(*random_view(), *random_view())
    torch.cuda.synchronize()
    seconds_per_step = (time.time() - start) / (N_STEPS - 2)
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    reserved_gib = torch.cuda.max_memory_reserved() / 1024**3
    print(f"RESULT windowsize={args.windowsize} batchsize={args.batchsize} peak_allocated={peak_gib:.1f}GiB peak_reserved={reserved_gib:.1f}GiB s_per_step={seconds_per_step:.2f}")
except torch.cuda.OutOfMemoryError:
    print(f"RESULT windowsize={args.windowsize} batchsize={args.batchsize} OOM")

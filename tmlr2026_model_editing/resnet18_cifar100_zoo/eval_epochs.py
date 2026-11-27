"""Evaluate zoo checkpoints (all epochs) on the CIFAR-100 test set; store per-image predictions.
usage: eval_epochs.py OUTDIR SHARD NSHARDS [--pilot NMODELS]"""
import os, sys, io, json, time, re, glob
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
TAR = "/path/to/zoos/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform.tar"
EXT = "/path/to/zoos/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform"
NPZ = "/path/to/scratch/sane_zoo/cifar100_resnet18/cifar100.npz"
OUT, SHARD, NSH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
PILOT = int(sys.argv[sys.argv.index("--pilot") + 1]) if "--pilot" in sys.argv else 0
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
dev = torch.device("cuda")

class Block(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.conv1 = nn.Conv2d(i, o, 3, st, 1, bias=False); s.bn1 = nn.BatchNorm2d(o)
        s.conv2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.bn2 = nn.BatchNorm2d(o)
        s.downsample = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o)) if (st != 1 or i != o) else None
    def forward(s, x):
        h = F.relu(s.bn1(s.conv1(x))); h = s.bn2(s.conv2(h))
        return F.relu(h + (s.downsample(x) if s.downsample is not None else x))
class ResNet18(nn.Module):
    def __init__(s, nc=100):
        super().__init__()
        s.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False); s.bn1 = nn.BatchNorm2d(64)
        cfg = [(64, 64, 1), (64, 128, 2), (128, 256, 2), (256, 512, 2)]
        for li, (i, o, st) in enumerate(cfg, 1):
            setattr(s, f"layer{li}", nn.Sequential(Block(i, o, st), Block(o, o, 1)))
        s.fc = nn.Linear(512, nc)
    def forward(s, x):
        h = F.relu(s.bn1(s.conv1(x)))
        h = s.layer4(s.layer3(s.layer2(s.layer1(h))))
        return s.fc(F.adaptive_avg_pool2d(h, 1).flatten(1))

d = np.load(NPZ); xte, yte = d["xte"], d["yte"]
mean = torch.tensor((0.5071, 0.4865, 0.4409)).view(1, 3, 1, 1); std = torch.tensor((0.2673, 0.2564, 0.2762)).view(1, 3, 1, 1)
X = ((torch.as_tensor(xte).permute(0, 3, 1, 2).float() / 255 - mean) / std).to(dev).contiguous(memory_format=torch.channels_last)
Y = torch.as_tensor(yte).long().to(dev)
assert X.shape == (10000, 3, 32, 32) and torch.bincount(Y).min() == 100

names = sorted(os.path.basename(p) for p in glob.glob(EXT + "/NN_tune_trainable_4008a_*"))
assert len(names) == 1000
if PILOT:
    models = list(range(0, 1000, 1000 // PILOT))[:PILOT]; epochs = list(range(0, 61, 5)); index = None
else:  # extracted checkpoints only: epochs 0, 5, ..., 60
    models = list(range(SHARD, 1000, NSH)); epochs = list(range(0, 61, 5)); index = None
E = 61
tag = f"pilot" if PILOT else f"shard{SHARD}of{NSH}"
fp = os.path.join(OUT, f"preds_{tag}.npy"); fd = os.path.join(OUT, f"done_{tag}.npy")
if os.path.exists(fp) and os.path.exists(fd):
    preds = np.load(fp); done = np.load(fd)
else:
    preds = np.full((len(models), E, 10000), 255, np.uint8); done = np.zeros((len(models), E), bool)
print(tag, "models", len(models), "epochs", len(epochs), "already done", int(done.sum()), flush=True)

def load(job):
    mi, e = job; name = names[models[mi]]
    if index is None:
        b = open(f"{EXT}/{name}/checkpoint_{e:06d}/checkpoints", "rb").read()
    else:
        off, size = index[name]["checkpoints"][str(e)]
        with open(TAR, "rb") as f:
            f.seek(off); b = f.read(size)
        assert len(b) == size
    return job, torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)

net = ResNet18().to(dev).eval().to(memory_format=torch.channels_last)
jobs = [(mi, e) for mi in range(len(models)) for e in epochs if not done[mi, e]]
t0 = time.time(); last_save = t0; n = 0
with ThreadPoolExecutor(8) as ex, torch.no_grad():
    for (mi, e), sd in ex.map(load, jobs):
        net.load_state_dict(sd, strict=True)
        p = torch.cat([net(X[i:i + 5000]).argmax(1) for i in range(0, 10000, 5000)])
        preds[mi, e] = p.to(torch.uint8).cpu().numpy(); done[mi, e] = True; n += 1
        if n % 100 == 0:
            print(f"{n}/{len(jobs)} {(time.time()-t0)/n:.3f} s/ckpt", flush=True)
        if time.time() - last_save > 600:
            np.save(fp, preds); np.save(fd, done); last_save = time.time()
np.save(fp, preds); np.save(fd, done)
corr = (preds == yte[None, None, :].astype(np.uint8)) & done[:, :, None]
np.savez(os.path.join(OUT, f"correct_{tag}.npz"), correct_bits=np.packbits(corr, axis=-1), done=done, models=np.array(models))
print("FINISHED", tag, n, "ckpts in", round(time.time() - t0), "s", flush=True)

# check against stored recall when available
sm = os.path.join(OUT, "stored_metrics.npz")
if True:
    if os.path.exists(sm):
        st = np.load(sm)["recall"][models]
    else:
        st = np.full((len(models), E, 100), np.nan)
        for i, mi in enumerate(models):
            for line in open(f"{EXT}/{names[mi]}/result.json"):
                L = json.loads(line); st[i, L["training_iteration"]] = [L[f"acc_class_{c}"] for c in range(100)]
    cnt = np.bincount(yte, minlength=100)
    rec = np.stack([np.stack([np.bincount(yte[corr[i, e]], minlength=100) / cnt for e in range(E)]) for i in range(len(models))])
    m = done
    diff = np.abs(rec - st)[m]
    print("recall vs stored: max abs diff", float(diff.max()), "mean", float(diff.mean()), "frac cells exact", float((diff < 1e-6).mean()), flush=True)

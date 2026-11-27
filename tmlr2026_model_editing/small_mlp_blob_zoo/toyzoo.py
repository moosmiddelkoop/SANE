"""Toy playground for weight-space model editing.

Synthetic 2-D Gaussian-mixture classification task, tiny MLP classifiers trained
thousands-at-a-time with a leading model dimension, flatten/unflatten helpers,
and per-class recall evaluation.
"""
import math
import numpy as np
import torch

K = 6            # classes
D_IN = 2
H = 16           # hidden width
LAYERS = [(D_IN, H), (H, H), (H, K)]
N_PARAMS = sum(i * o + o for i, o in LAYERS)  # 405

# ---------------------------------------------------------------- task
# Class geometry: (0,1) entangled pair, (2,3) entangled pair, 4 and 5 isolated.
MEANS = np.array([
    [-2.0,  2.0], [-1.0,  2.0],      # pair A (close)
    [ 2.0,  2.0], [ 2.0,  1.0],      # pair B (close)
    [ 0.0, -2.5],                    # isolated
    [-3.0, -1.0],                    # isolated
], dtype=np.float32)
SIGMA = 0.45


import os as _os
TASK = _os.environ.get("TOYTASK", "blobs")      # 'blobs' | 'rings' | 'spirals'
RING_R = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)   # 6 concentric annuli
RING_SD = 0.12
SPIRAL_SD = 0.12
SPIRAL_TURNS = 1.25


def sample_points(rng, labels):
    """Draw one 2-D point per label for the active task (vectorised over labels)."""
    labels = np.asarray(labels)
    if TASK == "blobs":
        return rng.normal(MEANS[labels], SIGMA).astype(np.float32)
    n = len(labels)
    if TASK == "rings":
        r = RING_R[labels] + rng.normal(0.0, RING_SD, n)
        a = rng.uniform(0.0, 2 * np.pi, n)
        return np.stack([r * np.cos(a), r * np.sin(a)], 1).astype(np.float32)
    if TASK == "spirals":
        # 3 interleaved arms; each arm split into an inner (even class) and outer (odd class) half
        arm, half = labels // 2, labels % 2
        s = (half + rng.uniform(0.0, 1.0, n)) / 2.0          # position along arm in [0,1]
        r = 0.3 + 2.7 * s
        a = arm * 2 * np.pi / 3 + s * SPIRAL_TURNS * 2 * np.pi
        xy = np.stack([r * np.cos(a), r * np.sin(a)], 1) + rng.normal(0.0, SPIRAL_SD, (n, 2))
        return xy.astype(np.float32)
    raise ValueError(TASK)


def sample_class(rng, c, n):
    if TASK == "blobs":
        return rng.normal(MEANS[c], SIGMA, size=(n, D_IN)).astype(np.float32)
    return sample_points(rng, np.full(n, c))


def make_test_set(n_per_class=500, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([sample_class(rng, c, n_per_class) for c in range(K)])
    y = np.repeat(np.arange(K), n_per_class)
    return torch.tensor(X), torch.tensor(y)


def make_train_sets(M, n_train, rng, alpha=1.0, noise_max=0.2):
    """Per-model training sets with Dirichlet class proportions + label noise.
    Returns X (M,n,2), y (M,n), props (M,K), noise (M,)"""
    props = rng.dirichlet(alpha * np.ones(K), size=M).astype(np.float32)
    noise = rng.uniform(0.0, noise_max, size=M).astype(np.float32)
    X = np.empty((M, n_train, D_IN), np.float32)
    y = np.empty((M, n_train), np.int64)
    for m in range(M):
        labels = rng.choice(K, size=n_train, p=props[m])
        X[m] = sample_points(rng, labels)
        flip = rng.random(n_train) < noise[m]
        yn = labels.copy()
        yn[flip] = rng.integers(0, K, size=flip.sum())
        y[m] = yn
    return torch.tensor(X), torch.tensor(y), props, noise

# ---------------------------------------------------------------- params
def init_params(M, init_scale, gen):
    """init_scale: (M,) multiplier on Kaiming-uniform-like init. Returns list of tensors."""
    ps = []
    for i, o in LAYERS:
        bound = 1.0 / math.sqrt(i)
        W = (torch.rand(M, i, o, generator=gen) * 2 - 1) * bound * init_scale[:, None, None]
        b = (torch.rand(M, o, generator=gen) * 2 - 1) * bound * init_scale[:, None]
        ps += [W, b]
    return ps


def flatten(ps):
    M = ps[0].shape[0]
    return torch.cat([p.reshape(M, -1) for p in ps], dim=1)


def unflatten(theta):
    """theta (M, N_PARAMS) -> list of batched tensors."""
    M = theta.shape[0]
    ps, k = [], 0
    for i, o in LAYERS:
        ps.append(theta[:, k:k + i * o].reshape(M, i, o)); k += i * o
        ps.append(theta[:, k:k + o].reshape(M, o)); k += o
    return ps


def layer_slices():
    """name -> slice into the flat vector."""
    out, k = {}, 0
    for li, (i, o) in enumerate(LAYERS):
        out[f"W{li}"] = slice(k, k + i * o); k += i * o
        out[f"b{li}"] = slice(k, k + o); k += o
    return out


def forward(ps, X):
    """ps: batched params; X: (M,n,2) or (n,2) shared. Returns logits (M,n,K)."""
    h = X if X.dim() == 3 else X.unsqueeze(0).expand(ps[0].shape[0], -1, -1)
    n_layers = len(ps) // 2
    for l in range(n_layers):
        W, b = ps[2 * l], ps[2 * l + 1]
        h = torch.bmm(h, W) + b[:, None, :]
        if l < n_layers - 1:
            h = torch.relu(h)
    return h


def forward_flat(theta, X):
    return forward(unflatten(theta), X)


@torch.no_grad()
def per_class_recall(theta, Xte, yte, chunk=2000):
    """theta (M,N) -> recall (M,K), accuracy (M,)"""
    recs, accs = [], []
    for s in range(0, theta.shape[0], chunk):
        pred = forward_flat(theta[s:s + chunk], Xte).argmax(-1)          # (m,n)
        correct = (pred == yte[None, :]).float()
        recs.append(torch.stack([correct[:, yte == c].mean(1) for c in range(K)], 1))
        accs.append(correct.mean(1))
    return torch.cat(recs), torch.cat(accs)

# ---------------------------------------------------------------- training
def train_zoo(M, seed, steps=600, n_train=256, batch=64, ckpt_fracs=(0.08, 0.3, 1.0),
              lr_range=(-3.5, -0.7), wd_range=(0.0, 1e-2), init_range=(0.3, 3.0),
              alpha=1.0, noise_max=0.2, verbose=True):
    """Train M MLPs in parallel with per-model hyperparameters (Adam, manual batched).
    Returns dict of numpy arrays: theta (C,M,N), recall (C,M,K), acc (C,M), hp (M,...)."""
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    lr = torch.tensor(10 ** rng.uniform(*lr_range, size=M), dtype=torch.float32)
    wd = torch.tensor(rng.uniform(*wd_range, size=M), dtype=torch.float32)
    init_scale = torch.tensor(rng.uniform(*init_range, size=M), dtype=torch.float32)
    X, y, props, noise = make_train_sets(M, n_train, rng, alpha, noise_max)
    Xte, yte = make_test_set()

    ps = init_params(M, init_scale, gen)
    theta = flatten(ps).clone().requires_grad_(True)
    m1 = torch.zeros_like(theta); m2 = torch.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    ckpt_steps = [max(1, int(round(f * steps))) for f in ckpt_fracs]
    out_theta, out_rec, out_acc = [], [], []
    ar = torch.arange(M)[:, None]
    for t in range(1, steps + 1):
        idx = torch.randint(0, n_train, (M, batch), generator=gen)
        xb, yb = X[ar, idx], y[ar, idx]
        logits = forward_flat(theta, xb)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, K), yb.reshape(-1), reduction="none")
        loss = loss.reshape(M, batch).mean(1).sum()
        g, = torch.autograd.grad(loss, theta)
        with torch.no_grad():
            g = g + wd[:, None] * theta
            m1.mul_(b1).add_(g, alpha=1 - b1)
            m2.mul_(b2).addcmul_(g, g, value=1 - b2)
            mhat = m1 / (1 - b1 ** t); vhat = m2 / (1 - b2 ** t)
            theta -= lr[:, None] * mhat / (vhat.sqrt() + eps)
        if t in ckpt_steps:
            rec, acc = per_class_recall(theta.detach(), Xte, yte)
            out_theta.append(theta.detach().clone().numpy())
            out_rec.append(rec.numpy()); out_acc.append(acc.numpy())
            if verbose:
                print(f"step {t}: mean acc {acc.mean():.3f}  recall std per class {rec.std(0).numpy().round(2)}")
    return dict(theta=np.stack(out_theta), recall=np.stack(out_rec), acc=np.stack(out_acc),
                lr=lr.numpy(), wd=wd.numpy(), init_scale=init_scale.numpy(),
                props=props, noise=noise, ckpt_steps=np.array(ckpt_steps))

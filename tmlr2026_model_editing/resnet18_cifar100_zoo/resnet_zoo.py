"""CIFAR-100 ResNet-18 zoo (modelzoos.cc layout) -- flat parameter vectors, SANE-style row tokens, functional evaluation.

Parameter vector theta (P = 11,220,132): all float parameters of the state dict in state-dict order
(conv/fc weights, BN gamma/beta, fc bias). BN running statistics (9,600 values) are kept separately per model.

Tokens: one row per output channel of every conv / fc layer; the row is [conv weights of that channel (fan_in*k*k),
gamma, beta of the following BatchNorm] (fc: [W_fc[c], b_c]); rows are cut into chunks of D_T = 289 (zero padded)
-> N_TOK = 39,304 tokens with positions [global index, layer id (0..20), output channel]. Standardisation is per
parameter tensor (mu/sigma over the train split) applied to theta before tokenisation, so tokenize/untokenize are exact
gathers and the whole map is differentiable.
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_T = 289
NC = 100
DEV = torch.device("cuda" if os.environ.get("ZOO_DEVICE", "cuda") == "cuda" and torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- architecture spec (state-dict order)
def _block_specs():
    specs = [("conv1", "bn1", (64, 3, 3, 3))]
    cin = 64
    for li, (cout, stride) in enumerate([(64, 1), (128, 2), (256, 2), (512, 2)], start=1):
        for bi in range(2):
            s = stride if bi == 0 else 1
            pre = f"layer{li}.{bi}"
            specs.append((f"{pre}.conv1", f"{pre}.bn1", (cout, cin if bi == 0 else cout, 3, 3)))
            specs.append((f"{pre}.conv2", f"{pre}.bn2", (cout, cout, 3, 3)))
            if bi == 0 and (s != 1 or cin != cout):
                specs.append((f"{pre}.downsample.0", f"{pre}.downsample.1", (cout, cin, 1, 1)))
        cin = cout
    return specs

CONV_SPECS = _block_specs()  # 20 conv layers (17 3x3 + 3 downsample)
# state-dict ordered list of float parameter tensors: (key, shape)
PARAM_SPECS = []
for cw, bn, shp in CONV_SPECS:
    PARAM_SPECS += [(cw + ".weight", shp), (bn + ".weight", (shp[0],)), (bn + ".bias", (shp[0],))]
PARAM_SPECS += [("fc.weight", (NC, 512)), ("fc.bias", (NC,))]
BUFFER_KEYS = []
for _, bn, shp in CONV_SPECS:
    BUFFER_KEYS += [(bn + ".running_mean", shp[0]), (bn + ".running_var", shp[0])]
P = sum(int(np.prod(s)) for _, s in PARAM_SPECS)
NB = sum(n for _, n in BUFFER_KEYS)
OFFSETS = {}
_o = 0
for k, s in PARAM_SPECS:
    OFFSETS[k] = (_o, _o + int(np.prod(s)), s)
    _o += int(np.prod(s))
assert _o == P
TENSOR_ID = np.zeros(P, dtype=np.int64)  # param index -> tensor id (for per-tensor standardisation)
for t, (k, s) in enumerate(PARAM_SPECS):
    a, b, _ = OFFSETS[k]
    TENSOR_ID[a:b] = t
N_TENSORS = len(PARAM_SPECS)

# row layout: layers = 20 convs (+ following BN) and fc
LAYER_KEYS = [(cw, bn, shp) for cw, bn, shp in CONV_SPECS] + [("fc", None, (NC, 512))]
N_LAYERS = len(LAYER_KEYS)  # 21


def _build_token_index():
    """IDX (N_TOK, D_T) int64 param indices (P = padding -> zero), POS (N_TOK, 3), LAYER_ID (N_TOK,),
    ROW_OF_TOKEN (N_TOK,), and TOKENS_OF_LAYER list of slices."""
    idx_rows, pos_rows, layer_rows, row_rows, tok_of_layer = [], [], [], [], []
    n = 0
    for l, (wk, bnk, shp) in enumerate(LAYER_KEYS):
        n_out = shp[0]
        fan = int(np.prod(shp[1:]))
        wa, _, _ = OFFSETS[wk + ".weight"]
        if bnk is None:
            ba, _, _ = OFFSETS["fc.bias"]
            extra = lambda r: [ba + r]
        else:
            ga, _, _ = OFFSETS[bnk + ".weight"]; bb, _, _ = OFFSETS[bnk + ".bias"]
            extra = lambda r: [ga + r, bb + r]
        row_len = fan + len(extra(0))
        n_chunks = int(np.ceil(row_len / D_T))
        start = n
        for r in range(n_out):
            row = np.concatenate([np.arange(wa + r * fan, wa + (r + 1) * fan), np.array(extra(r))])
            row = np.concatenate([row, np.full(n_chunks * D_T - row_len, P)])
            for c in range(n_chunks):
                idx_rows.append(row[c * D_T:(c + 1) * D_T]); pos_rows.append((n, l, r)); layer_rows.append(l); row_rows.append(r)
                n += 1
        tok_of_layer.append(slice(start, n))
    IDX = torch.from_numpy(np.stack(idx_rows)).long()
    POS = torch.tensor(pos_rows, dtype=torch.long)
    return IDX, POS, torch.tensor(layer_rows), torch.tensor(row_rows), tok_of_layer


IDX, POS, LAYER_ID, ROW_OF_TOKEN, TOKENS_OF_LAYER = _build_token_index()
N_TOK = IDX.shape[0]
MASK = IDX < P  # real entries
FC_TOKENS = TOKENS_OF_LAYER[-1]  # 200 tokens: class c -> tokens FC_TOKENS.start + 2c, +2c+1
TOK_PER_CLASS = (FC_TOKENS.stop - FC_TOKENS.start) // NC
_flat_pos = torch.full((P + 1,), -1, dtype=torch.long)
_flat_pos[IDX.reshape(-1)] = torch.arange(N_TOK * D_T)
INV = _flat_pos[:P].clone()  # param index -> position in flattened token array
assert (INV >= 0).all()


def tokenize(theta):
    """(B, P) -> (B, N_TOK, D_T). Differentiable gather; padding entries are 0."""
    th = torch.cat([theta, theta.new_zeros(theta.shape[0], 1)], 1)
    return th[:, IDX.to(theta.device)]


def untokenize(tok):
    """(B, N_TOK, D_T) -> (B, P); exact inverse of tokenize."""
    return tok.reshape(tok.shape[0], -1)[:, INV.to(tok.device)]


def tokenize_slice(theta, sl):
    """tokens sl (slice over token index) only: (B, len, D_T)."""
    th = torch.cat([theta, theta.new_zeros(theta.shape[0], 1)], 1)
    return th[:, IDX[sl].to(theta.device)]


def param_indices_of_tokens(sl):
    """unique real parameter indices covered by token slice sl (sorted LongTensor)."""
    ids = IDX[sl].reshape(-1)
    return torch.unique(ids[ids < P])


# ---------------------------------------------------------------- state dict <-> flat
def sd_to_flat(sd):
    th = np.empty(P, dtype=np.float32)
    for k, s in PARAM_SPECS:
        a, b, _ = OFFSETS[k]
        v = sd[k]
        assert tuple(v.shape) == tuple(s), (k, v.shape, s)
        th[a:b] = v.detach().cpu().numpy().reshape(-1)
    buf = np.empty(NB, dtype=np.float32); o = 0
    for k, n in BUFFER_KEYS:
        buf[o:o + n] = sd[k].detach().cpu().numpy().reshape(-1); o += n
    return th, buf


def flat_to_params(theta):
    """(B, P) -> dict key -> (B, *shape) views."""
    B = theta.shape[0]
    return {k: theta[:, a:b].reshape(B, *s) for k, (a, b, s) in OFFSETS.items()}


def buffers_to_dict(buf):
    out, o = {}, 0
    for k, n in BUFFER_KEYS:
        out[k] = buf[..., o:o + n]; o += n
    return out


# ---------------------------------------------------------------- functional ResNet-18 (CIFAR variant)
def resnet18_forward(theta, buf, x, maxpool=False, return_features=False):
    """theta (P,) flat params of ONE model, buf (NB,) its BN buffers, x (N,3,32,32) normalised images.
    BN in eval mode with running stats. Returns logits (N,100) [and GAP features (N,512)]."""
    p = flat_to_params(theta[None]); p = {k: v[0] for k, v in p.items()}
    b = buffers_to_dict(buf)

    def bn(h, k):
        return F.batch_norm(h, b[k + ".running_mean"], b[k + ".running_var"], p[k + ".weight"], p[k + ".bias"], training=False, eps=1e-5)

    h = F.relu(bn(F.conv2d(x, p["conv1.weight"], stride=1, padding=1), "bn1"))
    if maxpool:
        h = F.max_pool2d(h, 3, 2, 1)
    for li, stride in [(1, 1), (2, 2), (3, 2), (4, 2)]:
        for bi in range(2):
            s = stride if bi == 0 else 1
            pre = f"layer{li}.{bi}"
            out = F.relu(bn(F.conv2d(h, p[pre + ".conv1.weight"], stride=s, padding=1), pre + ".bn1"))
            out = bn(F.conv2d(out, p[pre + ".conv2.weight"], stride=1, padding=1), pre + ".bn2")
            if pre + ".downsample.0.weight" in p:
                sc = bn(F.conv2d(h, p[pre + ".downsample.0.weight"], stride=s), pre + ".downsample.1")
            else:
                sc = h
            h = F.relu(out + sc)
    feat = F.adaptive_avg_pool2d(h, 1).flatten(1)
    logits = feat @ p["fc.weight"].T + p["fc.bias"]
    return (logits, feat) if return_features else logits


# CIFAR-100 preprocessing candidates (verified in the Phase-0 job)
NORMS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "none": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}
NORM = os.environ.get("ZOO_NORM", "cifar100")


def prep_images(x_u8, norm=None):
    """uint8 (N,32,32,3) -> float (N,3,32,32) normalised."""
    m, s = NORMS[norm or NORM]
    x = torch.as_tensor(np.ascontiguousarray(x_u8)).permute(0, 3, 1, 2).float() / 255.0
    m = torch.tensor(m).view(1, 3, 1, 1); s = torch.tensor(s).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def predict(theta, buf, X, bs=1000, maxpool=False, features=False):
    """logits argmax (N,) for one model; X normalised images on the same device."""
    preds, feats = [], []
    for i in range(0, len(X), bs):
        out = resnet18_forward(theta, buf, X[i:i + bs], maxpool=maxpool, return_features=features)
        if features:
            preds.append(out[0].argmax(1)); feats.append(out[1])
        else:
            preds.append(out.argmax(1))
    return (torch.cat(preds), torch.cat(feats)) if features else torch.cat(preds)


@torch.no_grad()
def per_class_recall(theta, buf, X, y, bs=1000, maxpool=False):
    """theta (B,P), buf (B,NB), X (N,3,32,32), y (N,) -> recall (B,100), acc (B,)"""
    rec = torch.zeros(theta.shape[0], NC, device=X.device); acc = torch.zeros(theta.shape[0], device=X.device)
    counts = torch.bincount(y, minlength=NC).float()
    for i in range(theta.shape[0]):
        pr = predict(theta[i], buf[i], X, bs=bs, maxpool=maxpool)
        hit = (pr == y)
        rec[i] = torch.bincount(y[hit], minlength=NC).float() / counts.clamp(min=1)
        acc[i] = hit.float().mean()
    return rec, acc


def load_cifar100(path):
    d = np.load(path)
    return d["xtr"], d["ytr"], d["xte"], d["yte"]


# ---------------------------------------------------------------- zoo files (stage_zoo.py layout)
class Zoo:
    """zoo_theta.f32 (M,P) memmap, zoo_bn.npy (M,NB), zoo_meta.npz, zoo_tokens.f16 (M,N_TOK,D_T) standardised."""

    def __init__(self, root):
        self.root = root
        self.meta = np.load(os.path.join(root, "zoo_meta.npz"), allow_pickle=True)
        self.M = int(self.meta["n_models"])
        self.theta = np.memmap(os.path.join(root, "zoo_theta.f32"), dtype=np.float32, mode="r", shape=(self.M, P))
        self.bn = np.load(os.path.join(root, "zoo_bn.npy"), mmap_mode="r")
        tp = os.path.join(root, "zoo_tokens.f16")
        self.tokens = np.memmap(tp, dtype=np.float16, mode="r", shape=(self.M, N_TOK, D_T)) if os.path.exists(tp) else None
        st = np.load(os.path.join(root, "zoo_std.npz"))
        self.mu_t = torch.from_numpy(st["mu"]).float(); self.sigma_t = torch.from_numpy(st["sigma"]).float()  # per tensor
        self.mu = self.mu_t[torch.from_numpy(TENSOR_ID)]; self.sigma = self.sigma_t[torch.from_numpy(TENSOR_ID)]  # per param (P,)
        self.recall = self.meta["recall"].astype(np.float32); self.split = self.meta["split"]
        self.test_acc = self.meta["test_acc"].astype(np.float32)

    def rows(self, idx):
        return torch.from_numpy(np.ascontiguousarray(self.theta[np.asarray(idx)]))

    def bufs(self, idx):
        return torch.from_numpy(np.ascontiguousarray(self.bn[np.asarray(idx)]))

    def idx(self, split):
        return np.where(self.split == split)[0]

    def standardize(self, theta):
        return (theta - self.mu.to(theta.device)) / self.sigma.to(theta.device)

    def destandardize(self, theta_std):
        return theta_std * self.sigma.to(theta_std.device) + self.mu.to(theta_std.device)


def weight_statistics(theta):
    """SANE-paper style weight statistics s(W): per parameter tensor mean, std, |.| mean, and quantiles
    (1, 25, 50, 75, 99 %). theta (B,P) -> (B, N_TENSORS*8)."""
    feats = []
    for k, (a, b, s) in OFFSETS.items():
        w = theta[:, a:b]
        q = torch.quantile(w.float(), torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99], device=w.device), dim=1).T
        feats.append(torch.cat([w.mean(1, keepdim=True), w.std(1, keepdim=True), w.abs().mean(1, keepdim=True), q], 1))
    return torch.cat(feats, 1)


def fc_row_statistics(theta):
    """Per-class output-row statistics: (B, 100, 7) = [||W_c||, mean, std, max, min of W_c, b_c, ||W_c|| / mean_k ||W_k||]."""
    p = flat_to_params(theta)
    W, b = p["fc.weight"], p["fc.bias"]
    n = W.norm(dim=2)
    return torch.stack([n, W.mean(2), W.std(2), W.max(2).values, W.min(2).values, b, n / n.mean(1, keepdim=True)], 2)

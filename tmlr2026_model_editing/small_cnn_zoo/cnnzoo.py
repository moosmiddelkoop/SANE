"""Small CNN Zoo (Unterthiner et al. 2020) utilities: layout, batched torch forward, MNIST loading.

Architecture (train_network.py, google-research/dnn_predict_accuracy):
  3 x [Conv2D(16, 3x3, stride 2, padding VALID, activation relu|tanh)] -> GlobalAvgPool -> Dense(10)
  input: grayscale 28x28 scaled to [-1, 1]  (x/255*2-1)
Flat layout (layout.csv, TF kernel order (kh, kw, in, out)):
  [0:16]      conv1 bias      [16:160]     conv1 kernel (3,3,1,16)
  [160:176]   conv2 bias      [176:2480]   conv2 kernel (3,3,16,16)
  [2480:2496] conv3 bias      [2496:4800]  conv3 kernel (3,3,16,16)
  [4800:4810] dense bias      [4810:4970]  dense kernel (16,10)
"""
import os
os.environ.setdefault("KMP_AFFINITY", "disabled")
import numpy as np
import torch
import torch.nn.functional as F

D = 4970
H = 16
DEV = torch.device(os.environ.get("ZOO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
SEG = {  # name: (start, end, tf_shape)
    "c1b": (0, 16, (16,)), "c1w": (16, 160, (3, 3, 1, 16)),
    "c2b": (160, 176, (16,)), "c2w": (176, 2480, (3, 3, 16, 16)),
    "c3b": (2480, 2496, (16,)), "c3w": (2496, 4800, (3, 3, 16, 16)),
    "db": (4800, 4810, (10,)), "dw": (4810, 4970, (16, 10)),
}
ACT = {"relu": 0, "tanh": 1}


def unflatten(theta):
    """theta: (B, 4970) tensor -> dict of torch-layout tensors.
    conv weights -> (B, out, in, kh, kw); dense -> (B, out=10, in=16)."""
    p = {}
    for k, (s, e, shp) in SEG.items():
        t = theta[:, s:e].reshape(theta.shape[0], *shp)
        if k.endswith("w") and k.startswith("c"):
            t = t.permute(0, 4, 3, 1, 2)  # (B, out, in, kh, kw)
        elif k == "dw":
            t = t.permute(0, 2, 1)  # (B, 10, 16)
        p[k] = t
    return p


def _act(x, act):
    """act: (B,) int tensor, 0=relu 1=tanh. x: (B, ...)"""
    if act.ndim == 0:
        return torch.relu(x) if int(act) == 0 else torch.tanh(x)
    m = (act == 1).view(-1, *([1] * (x.ndim - 1)))
    return torch.where(m, torch.tanh(x), torch.relu(x))


def _gconv(x, w, b, B, P):
    """x: (B, P, Cin, h, w) ; w: (B, Cout, Cin, 3, 3) ; b: (B, Cout).  Grouped conv over models."""
    Cin = x.shape[2]
    Cout = w.shape[1]
    hh, ww = x.shape[-2:]
    xx = x.permute(1, 0, 2, 3, 4).reshape(P, B * Cin, hh, ww)
    y = F.conv2d(xx, w.reshape(B * Cout, Cin, 3, 3), b.reshape(B * Cout), stride=2, groups=B)
    return y.reshape(P, B, Cout, y.shape[-2], y.shape[-1]).permute(1, 0, 2, 3, 4)


def features(theta, x, act, shared_x=True):
    """Penultimate (global-average-pooled) features (B, P, 16)."""
    B = theta.shape[0]
    p = unflatten(theta)
    if shared_x:
        P, _, hh, ww = x.shape
        h = x.unsqueeze(0).expand(B, P, 1, hh, ww)
    else:
        P = x.shape[1]
        h = x
    act = torch.as_tensor(act, device=theta.device)
    if act.ndim == 1 and act.numel() != B:
        raise ValueError("act must be scalar or length B")
    h = _act(_gconv(h, p["c1w"], p["c1b"], B, P), act)
    h = _act(_gconv(h, p["c2w"], p["c2b"], B, P), act)
    h = _act(_gconv(h, p["c3w"], p["c3b"], B, P), act)
    return h.mean(dim=(-2, -1))  # (B, P, 16)


def head(theta, h):
    p = unflatten(theta)
    return torch.einsum("bpi,boi->bpo", h, p["dw"]) + p["db"].unsqueeze(1)


def forward(theta, x, act, shared_x=True):
    """Batched forward.
    theta: (B, 4970); act: (B,) int or scalar int.
    x: (P, 1, H, W) shared across models (shared_x=True) or (B, P, 1, H, W).
    returns logits (B, P, 10)."""
    return head(theta, features(theta, x, act, shared_x))


def prep_images(X_uint8, gray_mode=None):
    """uint8 (N,H,W) or (N,H,W,3) -> float32 tensor (N,1,H,W) in [-1,1].
    Follows dnn_predict_accuracy/train_network.py::_preprocess_batch: x/255 -> 2x-1 ->
    grayscale as the plain channel mean (CIFAR-10 / SVHN zoos)."""
    X = np.asarray(X_uint8)
    x = torch.as_tensor(X, dtype=torch.float32) / 255.0
    x = x * 2 - 1
    if X.ndim == 4:
        x = x.mean(-1)
    return x.unsqueeze(1)


DATASETS = {"mnist": "mnist.npz", "fashion_mnist": "fashion_mnist.npz",
            "cifar10": "cifar10.npz", "svhn_cropped": "svhn.npz"}


def load_images(dataset, root="."):
    """-> Xtr, ytr, Xte, yte (uint8 images, int labels) for a Small-CNN-Zoo dataset name."""
    z = np.load(os.path.join(root, DATASETS[dataset]))
    return z["Xtr"], z["ytr"].astype(np.int64), z["Xte"], z["yte"].astype(np.int64)


@torch.no_grad()
def per_class_recall(theta, act, X, y, chunk=2000, n_classes=10):
    """theta (B,4970), act (B,), X (N,1,28,28) float, y (N,) int -> recall (B,10), acc (B,)"""
    B = theta.shape[0]
    y = torch.as_tensor(y, device=theta.device)
    correct = torch.zeros(B, n_classes, device=theta.device)
    counts = torch.bincount(y, minlength=n_classes).float()
    if chunk is None:  # edit_cnn.true_recall passes chunk=None; batching only, result-neutral
        chunk = 500
    for i in range(0, X.shape[0], chunk):
        logits = forward(theta, X[i:i + chunk], act)  # (B,P,10)
        pred = logits.argmax(-1)
        hit = (pred == y[i:i + chunk].unsqueeze(0)).float()  # (B,P)
        correct.index_add_(1, y[i:i + chunk], hit)
    rec = correct / counts
    acc = correct.sum(1) / counts.sum()
    return rec, acc


def permute_channels(theta, perms):
    perms = [torch.as_tensor(p, device=theta.device) for p in perms]
    """Apply hidden-channel permutations. perms: list of 3 index tensors (16,) for conv1..conv3 outputs.
    theta: (B,4970) -> (B,4970), function-preserving."""
    p = unflatten(theta)
    p = {k: v.clone() for k, v in p.items()}
    p1, p2, p3 = perms
    p["c1w"] = p["c1w"][:, p1]; p["c1b"] = p["c1b"][:, p1]
    p["c2w"] = p["c2w"][:, :, p1][:, p2]; p["c2b"] = p["c2b"][:, p2]
    p["c3w"] = p["c3w"][:, :, p2][:, p3]; p["c3b"] = p["c3b"][:, p3]
    p["dw"] = p["dw"][:, :, p3]
    return flatten(p)


def flip_signs(theta, signs):
    """tanh sign-flip symmetry. signs: list of 3 tensors (16,) in {-1,+1} for conv1..conv3 outputs."""
    p = unflatten(theta)
    p = {k: v.clone() for k, v in p.items()}
    s1, s2, s3 = [torch.as_tensor(s, device=theta.device) for s in signs]
    p["c1w"] = p["c1w"] * s1.view(1, -1, 1, 1, 1); p["c1b"] = p["c1b"] * s1
    p["c2w"] = p["c2w"] * s1.view(1, 1, -1, 1, 1) * s2.view(1, -1, 1, 1, 1); p["c2b"] = p["c2b"] * s2
    p["c3w"] = p["c3w"] * s2.view(1, 1, -1, 1, 1) * s3.view(1, -1, 1, 1, 1); p["c3b"] = p["c3b"] * s3
    p["dw"] = p["dw"] * s3.view(1, 1, -1)
    return flatten(p)


def flatten(p):
    B = p["c1b"].shape[0]
    out = torch.empty(B, D, dtype=p["c1b"].dtype, device=p["c1b"].device)
    for k, (s, e, shp) in SEG.items():
        t = p[k]
        if k.endswith("w") and k.startswith("c"):
            t = t.permute(0, 3, 4, 2, 1)  # back to (B,kh,kw,in,out)
        elif k == "dw":
            t = t.permute(0, 2, 1)
        out[:, s:e] = t.reshape(B, -1)
    return out


def load_zoo(path="mnist_zoo_labelled.npz"):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_mnist(path="mnist.npz"):
    z = np.load(path)
    return z["Xtr"], z["ytr"], z["Xte"], z["yte"]

"""Metanet improvement experiments: weight-space preprocessing, symmetry-aware
architectures, functional probing, and loss choices for per-class recall prediction."""
import math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from toyzoo import unflatten, flatten, forward_flat, LAYERS, K, H

# ----------------------------------------------------------- symmetry transforms
def scale_normalize(theta, eps=1e-8):
    """ReLU positive-homogeneity canonicalisation (function preserving): rescale every
    hidden neuron so its incoming weights+bias have unit L2 norm; push the scale into the
    outgoing weights. Applied layer by layer."""
    ps = list(unflatten(theta))
    for l in range(len(LAYERS) - 1):
        W, b = ps[2 * l], ps[2 * l + 1]                      # (B,i,o), (B,o)
        a = torch.sqrt((W ** 2).sum(1) + b ** 2).clamp_min(eps)  # (B,o)
        ps[2 * l] = W / a[:, None, :]
        ps[2 * l + 1] = b / a
        ps[2 * (l + 1)] = ps[2 * (l + 1)] * a[:, :, None]
    return flatten(ps)


def canon_sort(theta):
    """Permutation canonicalisation (function preserving): sort the hidden units of each
    hidden layer by the L2 norm of their outgoing weights (descending)."""
    ps = list(unflatten(theta))
    for l in range(len(LAYERS) - 1):
        Wn = ps[2 * (l + 1)]                                  # (B,o,next)
        key = (Wn ** 2).sum(2)                                # (B,o)
        P = torch.argsort(key, dim=1, descending=True)
        W, b = ps[2 * l], ps[2 * l + 1]
        ps[2 * l] = torch.gather(W, 2, P[:, None, :].expand(-1, W.shape[1], -1))
        ps[2 * l + 1] = torch.gather(b, 1, P)
        ps[2 * (l + 1)] = torch.gather(Wn, 1, P[:, :, None].expand(-1, -1, Wn.shape[2]))
    return flatten(ps)


def permute_hidden(theta, gen=None):
    ps = list(unflatten(theta)); B = theta.shape[0]
    for l in range(len(LAYERS) - 1):
        h = LAYERS[l][1]
        P = torch.argsort(torch.rand(B, h, generator=gen), dim=1)
        W, b = ps[2 * l], ps[2 * l + 1]
        ps[2 * l] = torch.gather(W, 2, P[:, None, :].expand(-1, W.shape[1], -1))
        ps[2 * l + 1] = torch.gather(b, 1, P)
        Wn = ps[2 * (l + 1)]
        ps[2 * (l + 1)] = torch.gather(Wn, 1, P[:, :, None].expand(-1, -1, Wn.shape[2]))
    return flatten(ps)


def random_rescale(theta, gen=None, lo=0.5, hi=2.0):
    """Augmentation along the ReLU scale symmetry (function preserving)."""
    ps = list(unflatten(theta)); B = theta.shape[0]
    for l in range(len(LAYERS) - 1):
        o = LAYERS[l][1]
        a = torch.exp(torch.empty(B, o).uniform_(math.log(lo), math.log(hi), generator=gen))
        ps[2 * l] = ps[2 * l] * a[:, None, :]; ps[2 * l + 1] = ps[2 * l + 1] * a
        ps[2 * (l + 1)] = ps[2 * (l + 1)] / a[:, :, None]
    return flatten(ps)

# ----------------------------------------------------------- models
class Standardize(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)

    def forward(self, x):
        return (x - self.mu) / self.sd


class MLPMeta(nn.Module):
    """Paper-style MLP on the flat (optionally canonicalised) weight vector."""
    def __init__(self, n_in, hidden=256, n_hidden=5, dropout=0.03, pre=(), std=None):
        super().__init__()
        self.pre = list(pre)
        self.std = Standardize(*std) if std is not None else nn.Identity()
        layers, d = [], n_in
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]; d = hidden
        layers += [nn.Linear(d, K)]
        self.net = nn.Sequential(*layers)

    def logits(self, theta):
        for f in self.pre:
            theta = f(theta)
        return self.net(self.std(theta))

    def forward(self, theta):
        return torch.sigmoid(self.logits(theta))


def mlp(d_in, d, d_out, n=2):
    layers = []
    for i in range(n):
        layers += [nn.Linear(d_in if i == 0 else d, d), nn.ReLU()]
    layers += [nn.Linear(d, d_out)]
    return nn.Sequential(*layers)


class NeuronNet(nn.Module):
    """Exactly permutation-invariant metanet over hidden neurons of the 2-16-16-6 MLP.
    Layer-1 neurons -> embeddings; messages along W1 edges to layer-2 neurons (optionally
    a second, backward round); class-specific readout along W2 edges."""
    def __init__(self, d=64, rounds=2, pre=(), wscale=None):
        super().__init__()
        self.pre = list(pre); self.rounds = rounds
        self.ws = wscale if wscale is not None else [1.0] * 6   # per-tensor scale for inputs
        self.enc1 = mlp(3, d, d)
        self.enc2 = mlp(1 + K, d, d)
        self.msg_f_h = nn.Linear(d, d); self.msg_f_w = nn.Linear(1, d, bias=False); self.msg_f_o = mlp(d, d, d, 1)
        self.upd2 = mlp(2 * d, d, d)
        if rounds > 1:
            self.msg_b_h = nn.Linear(d, d); self.msg_b_w = nn.Linear(1, d, bias=False); self.msg_b_o = mlp(d, d, d, 1)
            self.upd1 = mlp(2 * d, d, d)
            self.msg_f2_h = nn.Linear(d, d); self.msg_f2_w = nn.Linear(1, d, bias=False); self.msg_f2_o = mlp(d, d, d, 1)
            self.upd2b = mlp(2 * d, d, d)
        self.ro_h = nn.Linear(d, d); self.ro_w = nn.Linear(1, d, bias=False); self.ro_o = mlp(d, d, d, 1)
        self.cls_emb = nn.Parameter(torch.randn(K, d) * 0.1)
        self.out = mlp(3 * d + 1, 2 * d, 1, 2)

    def _msg(self, hs, Wedge, lin_h, lin_w, out):
        # hs (B,n_src,d), Wedge (B,n_src,n_dst) -> mean over src of out(relu(lin_h h + lin_w w))
        z = torch.relu(lin_h(hs)[:, :, None, :] + lin_w(Wedge[..., None]))   # (B,s,t,d)
        return out(z).mean(1)                                                # (B,t,d)

    def logits(self, theta):
        for f in self.pre:
            theta = f(theta)
        W0, b0, W1, b1, W2, b2 = unflatten(theta)
        s = self.ws
        W0, b0, W1, b1, W2, b2 = W0 / s[0], b0 / s[1], W1 / s[2], b1 / s[3], W2 / s[4], b2 / s[5]
        h1 = self.enc1(torch.cat([W0.transpose(1, 2), b0[..., None]], -1))              # (B,16,d)
        e2 = self.enc2(torch.cat([b1[..., None], W2], -1))                               # (B,16,d)
        h2 = self.upd2(torch.cat([e2, self._msg(h1, W1, self.msg_f_h, self.msg_f_w, self.msg_f_o)], -1))
        if self.rounds > 1:
            h1 = self.upd1(torch.cat([h1, self._msg(h2, W1.transpose(1, 2), self.msg_b_h, self.msg_b_w, self.msg_b_o)], -1))
            h2 = self.upd2b(torch.cat([h2, self._msg(h1, W1, self.msg_f2_h, self.msg_f2_w, self.msg_f2_o)], -1))
        r = self._msg(h2, W2, self.ro_h, self.ro_w, self.ro_o)                          # (B,K,d)
        g = h2.mean(1, keepdim=True).expand(-1, K, -1)
        e = self.cls_emb[None].expand(theta.shape[0], -1, -1)
        return self.out(torch.cat([r, g, e, b2[..., None]], -1))[..., 0]

    def forward(self, theta):
        return torch.sigmoid(self.logits(theta))


class ProbeNet(nn.Module):
    """Functional metanet: run the classifier on P probe inputs (fixed grid or learned),
    feed its softmax outputs to an MLP. Invariant to every function-preserving symmetry."""
    def __init__(self, P=16, learn=True, grid=None, hidden=256, n_hidden=3, init_sd=2.0, seed=0, center=(-0.5, 0.5)):
        super().__init__()
        if grid is not None:
            pts = torch.tensor(grid, dtype=torch.float32)
        else:
            g = torch.Generator().manual_seed(seed)
            pts = torch.randn(P, 2, generator=g) * init_sd + torch.tensor(center, dtype=torch.float32)
        self.probes = nn.Parameter(pts, requires_grad=learn)
        P = len(pts)
        layers, d = [], 2 * P * K
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        layers += [nn.Linear(d, K)]
        self.net = nn.Sequential(*layers)

    def logits(self, theta):
        z = forward_flat(theta, self.probes)                                    # (B,P,K)
        feat = torch.cat([torch.softmax(z, -1), torch.log_softmax(z, -1).clamp_min(-20) / 10], -1)
        return self.net(feat.reshape(theta.shape[0], -1))

    def forward(self, theta):
        return torch.sigmoid(self.logits(theta))

# ----------------------------------------------------------- training
def evaluate(net, X, Y, chunk=4096):
    net.eval(); out = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            out.append(net(X[s:s + chunk]))
    P = torch.cat(out).numpy()
    return P, r2_score(Y.numpy(), P), r2_score(Y.numpy(), P, multioutput="raw_values")


def train(net, Xtr, Ytr, Xva, Yva, epochs=60, lr=1e-3, wd=1e-5, batch=256, seed=0,
          aug=(), loss="mse", eval_every=5, verbose=False, time_limit=None):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps_per_ep = math.ceil(len(Xtr) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_ep, pct_start=0.05)
    best, best_state, hist = -1e9, None, []
    t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr), generator=gen)
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]; xb, yb = Xtr[idx], Ytr[idx]
            for f in aug:
                xb = f(xb, gen)
            z = net.logits(xb)
            if loss == "bce":
                L = F.binary_cross_entropy_with_logits(z, yb)
            else:
                L = ((torch.sigmoid(z) - yb) ** 2).mean()
            opt.zero_grad(); L.backward(); opt.step(); sched.step()
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            _, r2, _ = evaluate(net, Xva, Yva)
            hist.append((ep + 1, r2, time.time() - t0))
            if verbose:
                print(f"  ep {ep+1} val R2 {r2:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if r2 > best:
                best, best_state = r2, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if time_limit and time.time() - t0 > time_limit:
            break
    net.load_state_dict(best_state); net.eval()
    return net, best, hist

"""Metanetworks for the Small CNN Zoo (MNIST). All nets take (theta (B,4970), act (B,) int{0 relu,1 tanh})
and return per-class recall predictions in [0,1] (10 outputs). `.logits()` returns pre-sigmoid values."""
import os, math, time
os.environ.setdefault("KMP_AFFINITY", "disabled")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cnnzoo
from cnnzoo import SEG, unflatten, forward as cnn_forward

K = 10
D = cnnzoo.D


def act_onehot(act):
    act = torch.as_tensor(act).long()
    return F.one_hot(act, 2).float()


# ----------------------------------------------------------------- augmentation (function preserving)
def aug_permute(theta, act, gen=None):
    """One random channel permutation per layer, shared across the batch (cheap, still full group coverage over steps)."""
    perms = [torch.randperm(16, generator=gen) for _ in range(3)]
    return cnnzoo.permute_channels(theta, perms)


def aug_signflip(theta, act, gen=None, p=0.5):
    """Sign flips for tanh models only (identity for relu rows)."""
    signs = [torch.where(torch.rand(16, generator=gen) < p, -1.0, 1.0) for _ in range(3)]
    flipped = cnnzoo.flip_signs(theta, signs)
    m = (torch.as_tensor(act) == 1).view(-1, 1)
    return torch.where(m, flipped, theta)


# ----------------------------------------------------------------- 1. flat MLP (paper recipe + variants)
class MLPMeta(nn.Module):
    def __init__(self, hidden=256, n_hidden=5, dropout=0.03, use_act=False, mu=None, sd=None):
        super().__init__()
        self.use_act = use_act
        self.register_buffer("mu", torch.zeros(D) if mu is None else torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.ones(D) if sd is None else torch.as_tensor(sd, dtype=torch.float32))
        d = D + (2 if use_act else 0)
        layers = []
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, K)]
        self.net = nn.Sequential(*layers)

    def logits(self, theta, act):
        x = (theta - self.mu) / self.sd
        if self.use_act:
            x = torch.cat([x, act_onehot(act)], -1)
        return self.net(x)

    def forward(self, theta, act):
        return torch.sigmoid(self.logits(theta, act))


# ----------------------------------------------------------------- 2. permutation-equivariant channel-graph net
def mlp(d_in, d, d_out, n=2):
    layers, cur = [], d_in
    for _ in range(n):
        layers += [nn.Linear(cur, d), nn.ReLU()]
        cur = d
    layers += [nn.Linear(cur, d_out)]
    return nn.Sequential(*layers)


class ChannelGraphNet(nn.Module):
    """Nodes = channels of conv1/conv2/conv3 (16 each); input node and 10 class nodes fixed.
    Edge features = flattened 3x3 kernels. Exactly invariant to channel permutations in every layer.
    rounds=1: forward message passing conv1->conv2->conv3->classes. rounds=2 adds a backward pass."""

    def __init__(self, d=64, rounds=1, scale=None):
        super().__init__()
        self.rounds = rounds
        s = scale if scale is not None else {k: 1.0 for k in SEG}
        self.register_buffer("scale_vec", torch.tensor([s[k] for k in ["c1w", "c1b", "c2w", "c2b", "c3w", "c3b", "dw", "db"]]))
        self.enc1 = mlp(9 + 1 + 2, d, d)        # conv1 kernel (in=1) + bias + act onehot
        self.enc2 = mlp(1 + 2, d, d)             # bias + act
        self.enc3 = mlp(1 + 2, d, d)
        self.m12_h = nn.Linear(d, d); self.m12_w = nn.Linear(9, d, bias=False); self.m12_o = mlp(d, d, d, 1)
        self.m23_h = nn.Linear(d, d); self.m23_w = nn.Linear(9, d, bias=False); self.m23_o = mlp(d, d, d, 1)
        self.upd2 = mlp(2 * d, d, d); self.upd3 = mlp(2 * d, d, d)
        if rounds > 1:
            self.b32_h = nn.Linear(d, d); self.b32_w = nn.Linear(9, d, bias=False); self.b32_o = mlp(d, d, d, 1)
            self.b21_h = nn.Linear(d, d); self.b21_w = nn.Linear(9, d, bias=False); self.b21_o = mlp(d, d, d, 1)
            self.updb2 = mlp(2 * d, d, d); self.updb1 = mlp(2 * d, d, d)
            self.f12_h = nn.Linear(d, d); self.f12_w = nn.Linear(9, d, bias=False); self.f12_o = mlp(d, d, d, 1)
            self.f23_h = nn.Linear(d, d); self.f23_w = nn.Linear(9, d, bias=False); self.f23_o = mlp(d, d, d, 1)
            self.updf2 = mlp(2 * d, d, d); self.updf3 = mlp(2 * d, d, d)
        self.ro_h = nn.Linear(d, d); self.ro_w = nn.Linear(1, d, bias=False); self.ro_o = mlp(d, d, d, 1)
        self.cls_emb = nn.Parameter(torch.randn(K, d) * 0.1)
        self.out = mlp(3 * d + 1 + 2, 2 * d, 1, 2)

    @staticmethod
    def _msg(hs, E, lin_h, lin_w, out):
        # hs (B,S,d) ; E (B,S,T,f) -> (B,T,d) mean over sources
        z = torch.relu(lin_h(hs)[:, :, None, :] + lin_w(E))
        return out(z).mean(1)

    def logits(self, theta, act):
        p = unflatten(theta)
        sv = self.scale_vec
        sq = torch.asinh   # squash scaled weights: linear near 0, log for large |w| (keeps extreme models finite)
        c1w, c1b, c2w, c2b, c3w, c3b, dw, db = (sq(p["c1w"] / sv[0]), sq(p["c1b"] / sv[1]), sq(p["c2w"] / sv[2]), sq(p["c2b"] / sv[3]),
                                                sq(p["c3w"] / sv[4]), sq(p["c3b"] / sv[5]), sq(p["dw"] / sv[6]), sq(p["db"] / sv[7]))
        B = theta.shape[0]
        a = act_onehot(act).to(theta.dtype)                     # (B,2)
        a16 = a[:, None, :].expand(B, 16, 2)
        # conv weights (B,out,in,3,3) -> edges E[b, in, out, 9]
        E12 = c2w.permute(0, 2, 1, 3, 4).reshape(B, 16, 16, 9)
        E23 = c3w.permute(0, 2, 1, 3, 4).reshape(B, 16, 16, 9)
        h1 = self.enc1(torch.cat([c1w.reshape(B, 16, 9), c1b[..., None], a16], -1))
        h2 = self.upd2(torch.cat([self.enc2(torch.cat([c2b[..., None], a16], -1)), self._msg(h1, E12, self.m12_h, self.m12_w, self.m12_o)], -1))
        h3 = self.upd3(torch.cat([self.enc3(torch.cat([c3b[..., None], a16], -1)), self._msg(h2, E23, self.m23_h, self.m23_w, self.m23_o)], -1))
        if self.rounds > 1:
            h2 = self.updb2(torch.cat([h2, self._msg(h3, E23.transpose(1, 2), self.b32_h, self.b32_w, self.b32_o)], -1))
            h1 = self.updb1(torch.cat([h1, self._msg(h2, E12.transpose(1, 2), self.b21_h, self.b21_w, self.b21_o)], -1))
            h2 = self.updf2(torch.cat([h2, self._msg(h1, E12, self.f12_h, self.f12_w, self.f12_o)], -1))
            h3 = self.updf3(torch.cat([h3, self._msg(h2, E23, self.f23_h, self.f23_w, self.f23_o)], -1))
        Eo = dw.permute(0, 2, 1)[..., None]                      # (B,16,K,1)
        r = self._msg(h3, Eo, self.ro_h, self.ro_w, self.ro_o)   # (B,K,d)
        g = h3.mean(1, keepdim=True).expand(-1, K, -1)
        e = self.cls_emb[None].expand(B, -1, -1)
        aK = a[:, None, :].expand(B, K, 2)
        return self.out(torch.cat([r, g, e, db[..., None], aK], -1))[..., 0]

    def forward(self, theta, act):
        return torch.sigmoid(self.logits(theta, act))


# ----------------------------------------------------------------- 3. probe nets
class ProbeNet(nn.Module):
    """Runs each classifier on P probe images, regresses recall from softmax / log-softmax outputs.
    mode: 'free'  - probes = tanh(u), u free, u ~ N(0, init_sd)
          'gen'   - probes = tanh(G(z)), deep linear generator G from fixed latents z
          'real'  - probes = tanh(u), u initialised from real images (passed via init_images, in [-1,1])
          'fixed' - like 'real' but probes are not trained."""

    def __init__(self, P=64, mode="free", hidden=256, n_hidden=3, init_sd=0.5, seed=0, init_images=None,
                 latent=64, use_act=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.mode, self.P, self.use_act = mode, P, use_act
        if mode in ("free",):
            self.u = nn.Parameter(torch.randn(P, 1, 28, 28, generator=g) * init_sd)
        elif mode in ("real", "fixed"):
            x = torch.as_tensor(init_images, dtype=torch.float32).reshape(P, 1, 28, 28).clamp(-0.999, 0.999)
            self.u = nn.Parameter(torch.atanh(x), requires_grad=(mode == "real"))
        elif mode == "gen":
            self.register_buffer("z", torch.randn(P, latent, generator=g))
            self.G = nn.Sequential(nn.Linear(latent, 256), nn.Linear(256, 784))
            with torch.no_grad():
                for m in self.G:
                    nn.init.normal_(m.weight, std=0.5 / math.sqrt(m.in_features)); nn.init.zeros_(m.bias)
        else:
            raise ValueError(mode)
        d = 2 * P * K + (2 if use_act else 0)
        layers = []
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, K)]
        self.net = nn.Sequential(*layers)

    def probes(self):
        if self.mode == "gen":
            return torch.tanh(self.G(self.z)).reshape(self.P, 1, 28, 28)
        return torch.tanh(self.u)

    def logits(self, theta, act):
        z = cnn_forward(theta, self.probes(), act)                     # (B,P,K)
        feat = torch.cat([torch.softmax(z, -1), torch.log_softmax(z, -1).clamp_min(-20) / 10], -1).reshape(theta.shape[0], -1)
        if self.use_act:
            feat = torch.cat([feat, act_onehot(act).to(feat.dtype)], -1)
        return self.net(feat)

    def forward(self, theta, act):
        return torch.sigmoid(self.logits(theta, act))


# ----------------------------------------------------------------- 4. invariant hand features (for ridge)
def invariant_features(theta, act):
    """Permutation-invariant summary features: per-tensor rms/mean/std/max|.|, sorted output weights per class, biases, act."""
    theta = torch.as_tensor(theta)
    p = unflatten(theta)
    B = theta.shape[0]
    feats = []
    for k in ["c1w", "c1b", "c2w", "c2b", "c3w", "c3b", "dw", "db"]:
        t = p[k].reshape(B, -1)
        feats += [t.pow(2).mean(1, keepdim=True).sqrt(), t.mean(1, keepdim=True), t.std(1, keepdim=True), t.abs().amax(1, keepdim=True)]
    dw = p["dw"]                                     # (B,K,16)
    feats.append(torch.sort(dw, dim=-1).values.reshape(B, -1))   # sorted incoming weights per class (K*16)
    feats.append(p["db"])                            # class biases
    feats.append(dw.norm(dim=-1))                    # per-class weight norm
    feats.append(act_onehot(act).to(theta.dtype))
    x = torch.cat(feats, -1)
    return torch.cat([x, torch.log1p(x.abs())], -1)


# ----------------------------------------------------------------- metrics
def r2(y, yh):
    y = np.asarray(y, dtype=np.float64); yh = np.asarray(yh, dtype=np.float64)
    return float(1 - ((y - yh) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def r2_cols(y, yh):
    y = np.asarray(y, dtype=np.float64); yh = np.asarray(yh, dtype=np.float64)
    return 1 - ((y - yh) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)


def centred(y):
    return y - y.mean(-1, keepdims=True)


def metric_table(Y, P, step, acc, act, prefix=""):
    """Y,P (N,10); step (N,) in {0,20,86}; acc (N,); act (N,). Returns dict of metrics."""
    out = {}
    out[prefix + "r2_all"] = r2(Y, P)
    out[prefix + "mae_all"] = float(np.abs(Y - P).mean())
    f = step == 86
    out[prefix + "r2_final"] = r2(Y[f], P[f])
    out[prefix + "mae_final"] = float(np.abs(Y[f] - P[f]).mean())
    t = f & (acc >= 0.5)
    out[prefix + "r2_centred_final_trained"] = r2(centred(Y[t]), centred(P[t]))
    out[prefix + "mae_centred_final_trained"] = float(np.abs(centred(Y[t]) - centred(P[t])).mean())
    out[prefix + "r2_final_trained"] = r2(Y[t], P[t])
    out[prefix + "n_final_trained"] = int(t.sum())
    for a, nm in [(0, "relu"), (1, "tanh")]:
        s = f & (act == a)
        out[prefix + f"r2_final_{nm}"] = r2(Y[s], P[s])
        s2 = t & (act == a)
        out[prefix + f"r2_centred_final_trained_{nm}"] = r2(centred(Y[s2]), centred(P[s2]))
    pc = r2_cols(Y[f], P[f])
    for c in range(K):
        out[prefix + f"r2_final_c{c}"] = float(pc[c])
    pcc = r2_cols(centred(Y[t]), centred(P[t]))
    for c in range(K):
        out[prefix + f"r2_centred_final_trained_c{c}"] = float(pcc[c])
    m = step == 20
    out[prefix + "r2_middle"] = r2(Y[m], P[m])
    return out


# ----------------------------------------------------------------- training
@torch.no_grad()
def predict(net, theta, act, chunk=1024):
    net.eval(); out = []
    dev = next(net.parameters()).device
    for s in range(0, len(theta), chunk):
        out.append(net(torch.as_tensor(theta[s:s + chunk]).to(dev), torch.as_tensor(act[s:s + chunk]).to(dev)))
    return torch.cat(out).cpu().numpy()


def train(net, Xtr, Atr, Ytr, Xva, Ava, Yva, epochs=30, lr=1e-3, wd=1e-5, batch=256, seed=0, aug=(),
          eval_every=1, verbose=True, time_limit=None, log=None, clip=None):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    dev = cnnzoo.DEV; net.to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps_per_ep = math.ceil(len(Xtr) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_ep, pct_start=0.05)
    best, best_state, hist = -1e9, None, []
    t0 = time.time()
    Ytr_t = torch.as_tensor(Ytr); Atr_t = torch.as_tensor(Atr)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(Xtr), generator=gen)
        tot = 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            xb = torch.as_tensor(Xtr[np.sort(idx.numpy())]) if isinstance(Xtr, np.memmap) else torch.as_tensor(Xtr[idx])
            if isinstance(Xtr, np.memmap):
                idx = torch.as_tensor(np.sort(idx.numpy()))
            ab, yb = Atr_t[idx].to(dev), Ytr_t[idx].to(dev)
            xb = xb.to(dev)
            for f in aug:
                xb = f(xb, ab, gen)
            z = net.logits(xb, ab)
            L = ((torch.sigmoid(z) - yb) ** 2).mean()
            opt.zero_grad(); L.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step(); sched.step()
            tot += L.item() * len(idx)
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            pv = predict(net, Xva, Ava)
            r2v = r2(Yva, pv)
            hist.append({"epoch": ep + 1, "train_mse": tot / len(Xtr), "val_r2": r2v, "time": time.time() - t0})
            if verbose:
                print(f"  ep {ep+1} train mse {tot/len(Xtr):.5f} val R2 {r2v:.4f} ({time.time()-t0:.0f}s)", flush=True)
            if log:
                import json
                json.dump(hist, open(log, "w"))
            if r2v > best:
                best, best_state = r2v, {k: v.detach().clone() for k, v in net.state_dict().items()}
        if time_limit and time.time() - t0 > time_limit:
            break
    net.load_state_dict(best_state); net.eval()
    return net, best, hist

"""SANE (Schürholt, Mahoney, Borth, ICML 2024) adapted to the Unterthiner Small CNN Zoo.

Tokenisation: one token per output channel / output unit, bias appended (as in the SANE
repository's tokenize_checkpoint), zero-padded to D_T = 145:
  layer 0 conv1: 16 tokens of 9 weights + 1 bias
  layer 1 conv2: 16 tokens of 144 + 1
  layer 2 conv3: 16 tokens of 144 + 1
  layer 3 dense: 10 tokens of 16 + 1          -> 58 tokens, 3-d positions [n, layer, row]
Standardisation: scalar per-layer mean / std over (weights+bias) of the train split, repo style.
Alignment: git-re-basin weight matching of the three hidden-channel permutations to one
reference model per zoo (tanh sign flips are NOT handled, as in SANE).
The AE itself is the vendored SANE implementation (sane_vendor/def_AE.py, MIT).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

import cnnzoo
from sane_vendor.def_AE import AE

D_T = 145
N_TOK = 58
H = cnnzoo.H
# (weight key, bias key, n_out, fan_in*k*k)
LAYERS = [("c1w", "c1b", 16, 9), ("c2w", "c2b", 16, 144), ("c3w", "c3b", 16, 144), ("dw", "db", 10, 16)]
DENSE_TOKENS = slice(48, 58)  # the 10 per-class output rows


def token_mask_pos():
    """mask (N_TOK, D_T) bool of real entries; pos (N_TOK, 3) long [global, layer, row]; layer_id (N_TOK,)."""
    mask = torch.zeros(N_TOK, D_T, dtype=torch.bool)
    pos = torch.zeros(N_TOK, 3, dtype=torch.long)
    layer_id = torch.zeros(N_TOK, dtype=torch.long)
    n = 0
    for l, (_, _, n_out, fan) in enumerate(LAYERS):
        for r in range(n_out):
            mask[n, : fan + 1] = True
            pos[n] = torch.tensor([n, l, r])
            layer_id[n] = l
            n += 1
    assert n == N_TOK
    return mask, pos, layer_id


MASK, POS, LAYER_ID = token_mask_pos()


def tokenize(theta):
    """(B, 4970) -> (B, 58, 145) row tokens with bias appended, zero padded. Differentiable."""
    p = cnnzoo.unflatten(theta)
    B = theta.shape[0]
    toks = []
    for wk, bk, n_out, fan in LAYERS:
        w = p[wk].reshape(B, n_out, fan)
        b = p[bk].reshape(B, n_out, 1)
        t = torch.cat([w, b], dim=2)
        toks.append(F.pad(t, (0, D_T - fan - 1)))
    return torch.cat(toks, dim=1)


def untokenize(tok):
    """(B, 58, 145) -> (B, 4970); exact inverse of tokenize (padding ignored)."""
    B = tok.shape[0]
    p = {}
    n = 0
    for wk, bk, n_out, fan in LAYERS:
        t = tok[:, n : n + n_out]
        n += n_out
        w = t[:, :, :fan]
        b = t[:, :, fan]
        if wk == "dw":
            p[wk] = w.reshape(B, 10, 16)
        else:
            p[wk] = w.reshape(B, n_out, fan // 9, 3, 3)
        p[bk] = b
    return cnnzoo.flatten(p)


# ---------------------------------------------------------------- standardisation
def layer_stats(theta_train):
    """Repo-style per-layer scalar statistics: mu = mean of per-model means, sigma = RMS of per-model stds
    over (weights + bias) of each layer. theta_train: (M, 4970) tensor. Returns mu (4,), sigma (4,)."""
    tok = tokenize(theta_train)
    mu = torch.zeros(len(LAYERS))
    sigma = torch.zeros(len(LAYERS))
    for l, (_, _, n_out, fan) in enumerate(LAYERS):
        rows = tok[:, LAYER_ID == l, : fan + 1].reshape(tok.shape[0], -1)
        mu[l] = rows.mean(dim=1).mean()
        sigma[l] = rows.std(dim=1).pow(2).mean().sqrt()
    return mu, sigma


def standardize(tok, mu, sigma):
    m = mu[LAYER_ID.to(tok.device)].view(1, N_TOK, 1)
    s = sigma[LAYER_ID.to(tok.device)].view(1, N_TOK, 1)
    return (tok - m) / s * MASK.to(tok.device)


def destandardize(tok_std, mu, sigma):
    m = mu[LAYER_ID.to(tok_std.device)].view(1, N_TOK, 1)
    s = sigma[LAYER_ID.to(tok_std.device)].view(1, N_TOK, 1)
    return (tok_std * s + m) * MASK.to(tok_std.device)


# ---------------------------------------------------------------- permutations
def permute_batched(theta, P):
    """Per-model hidden-channel permutations. theta (B,4970); P (B,3,16) long with
    aligned_channel[j] = original_channel[P[:, l, j]]. Differentiable in theta (gather)."""
    p = cnnzoo.unflatten(theta)
    p1, p2, p3 = P[:, 0], P[:, 1], P[:, 2]

    def g(t, idx, dim):
        shape = [1] * t.dim()
        shape[0], shape[dim] = t.shape[0], H
        return t.gather(dim, idx.view(shape).expand_as(t))

    out = {}
    out["c1w"] = g(p["c1w"], p1, 1); out["c1b"] = g(p["c1b"], p1, 1)
    out["c2w"] = g(g(p["c2w"], p1, 2), p2, 1); out["c2b"] = g(p["c2b"], p2, 1)
    out["c3w"] = g(g(p["c3w"], p2, 2), p3, 1); out["c3b"] = g(p["c3b"], p3, 1)
    out["dw"] = g(p["dw"], p3, 2); out["db"] = p["db"]
    return cnnzoo.flatten(out)


def invert_perms(P):
    inv = torch.empty_like(P)
    ar = torch.arange(H, device=P.device).expand_as(P)
    inv.scatter_(2, P, ar)
    return inv


def identity_perms(B, device="cpu"):
    return torch.arange(H, device=device).view(1, 1, H).expand(B, 3, H).clone()


def _rows(theta):
    """Per-layer row matrices incl. bias: W1 (B,16,10), W2 (B,16,16,9)+b2 (B,16), W3, Wd (B,10,16)."""
    p = cnnzoo.unflatten(theta)
    B = theta.shape[0]
    return dict(
        w1=torch.cat([p["c1w"].reshape(B, 16, 9), p["c1b"].reshape(B, 16, 1)], 2),
        w2=p["c2w"].reshape(B, 16, 16, 9), b2=p["c2b"],
        w3=p["c3w"].reshape(B, 16, 16, 9), b3=p["c3b"],
        wd=p["dw"],  # (B,10,16)
    )


def weight_match(theta, theta_ref, max_iter=20, seed=0):
    """Git-re-basin weight matching of each model in theta (B,4970) to theta_ref (4970,).
    Returns P (B,3,16) long such that permute_batched(theta, P) is aligned to theta_ref,
    and n_iter (B,) iterations until convergence."""
    B = theta.shape[0]
    dev = theta.device
    r = _rows(theta)
    q = _rows(theta_ref[None])
    P = identity_perms(B, dev)
    rng = np.random.default_rng(seed)
    n_iter = torch.zeros(B, dtype=torch.long)
    active = np.ones(B, dtype=bool)
    for it in range(max_iter):
        changed = np.zeros(B, dtype=bool)
        for l in rng.permutation(3):
            p1, p2, p3 = P[:, 0], P[:, 1], P[:, 2]
            if l == 0:
                # C[b, j(ref), i(model)] = <W1[i], W1ref[j]> + sum_o <W2[p2[o], i, :], W2ref[o, j, :]>
                C = torch.einsum("bik,jk->bji", r["w1"], q["w1"][0])
                w2p = r["w2"].gather(1, p2.view(B, H, 1, 1).expand(B, H, H, 9))  # rows permuted by p2
                C = C + torch.einsum("boik,ojk->bji", w2p, q["w2"][0])
            elif l == 1:
                w2c = r["w2"].gather(2, p1.view(B, 1, H, 1).expand(B, H, H, 9))  # columns permuted by p1
                C = torch.einsum("bik,jk->bji", torch.cat([w2c.reshape(B, H, -1), r["b2"][:, :, None]], 2),
                                 torch.cat([q["w2"][0].reshape(H, -1), q["b2"][0][:, None]], 1))
                w3p = r["w3"].gather(1, p3.view(B, H, 1, 1).expand(B, H, H, 9))
                C = C + torch.einsum("boik,ojk->bji", w3p, q["w3"][0])
            else:
                w3c = r["w3"].gather(2, p2.view(B, 1, H, 1).expand(B, H, H, 9))
                C = torch.einsum("bik,jk->bji", torch.cat([w3c.reshape(B, H, -1), r["b3"][:, :, None]], 2),
                                 torch.cat([q["w3"][0].reshape(H, -1), q["b3"][0][:, None]], 1))
                C = C + torch.einsum("boi,oj->bji", r["wd"], q["wd"][0])
            Cn = C.detach().cpu().numpy()
            for b in np.flatnonzero(active):
                _, col = linear_sum_assignment(Cn[b], maximize=True)
                new = torch.as_tensor(col, device=dev)
                if not torch.equal(new, P[b, l]):
                    P[b, l] = new
                    changed[b] = True
        n_iter[torch.as_tensor(active)] = it + 1
        active = active & changed
        if not active.any():
            break
    return P, n_iter


def weight_match_restarts(theta, theta_ref, restarts=8, seed=0, chunk=1000):
    """weight_match from the identity plus (restarts-1) random initial permutations; keeps the best
    objective per model (coordinate-descent weight matching has many local optima on a heterogeneous zoo).
    Returns P (B,3,16), objective (B,)."""
    outP, outO = [], []
    for s in range(0, theta.shape[0], chunk):
        th = theta[s : s + chunk]
        B = th.shape[0]
        gen = torch.Generator().manual_seed(seed + s)
        bestP, bestobj = None, None
        for r in range(restarts):
            if r == 0:
                P0 = identity_perms(B, th.device)
            else:
                P0 = torch.stack([torch.stack([torch.randperm(H, generator=gen) for _ in range(3)]) for _ in range(B)]).to(th.device)
            P, _ = weight_match(permute_batched(th, P0), theta_ref, seed=seed + r)
            Pc = P0.gather(2, P)  # composed permutation
            obj = alignment_objective(permute_batched(th, Pc), theta_ref)
            if bestP is None:
                bestP, bestobj = Pc, obj
            else:
                better = obj > bestobj
                bestP[better] = Pc[better]
                bestobj = torch.where(better, obj, bestobj)
        outP.append(bestP); outO.append(bestobj)
    return torch.cat(outP), torch.cat(outO)


def alignment_objective(theta, theta_ref):
    """<theta, theta_ref> restricted to permutation-sensitive tensors (for alignment diagnostics)."""
    r = _rows(theta); q = _rows(theta_ref[None])
    return sum((r[k] * q[k]).flatten(1).sum(1) for k in ["w1", "w2", "b2", "w3", "b3", "wd"])


def choose_reference(acc_train, idx_train):
    """Reference model = train-split model with the highest epoch-86 accuracy."""
    return int(idx_train[np.argmax(acc_train)])


# ---------------------------------------------------------------- SANE model wrapper
def ae_config(d_model=512, nhead=8, num_layers=4, lat_dim=128, dropout=0.0):
    return {"ae:i_dim": D_T, "ae:d_model": d_model, "ae:nhead": nhead, "ae:num_layers": num_layers,
            "ae:lat_dim": lat_dim, "training::windowsize": N_TOK, "ae:dropout": dropout,
            "ae:max_positions": [N_TOK, len(LAYERS), H], "ae:transformer_type": "gpt2"}


class SaneZoo(nn.Module):
    """Vendored SANE AE plus zoo-specific tokenisation / standardisation. Alignment is applied by the caller
    (permute_batched) so that the permutation can be fixed along an edit path."""

    def __init__(self, cfg, mu, sigma):
        super().__init__()
        self.cfg = cfg
        self.ae = AE(cfg)
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sigma", torch.as_tensor(sigma, dtype=torch.float32))
        self.register_buffer("pos", POS.clone())
        self.register_buffer("mask", MASK.clone())

    def tokens(self, theta_aligned):
        return standardize(tokenize(theta_aligned), self.mu, self.sigma)

    def _pos(self, B):
        return self.pos[None].expand(B, N_TOK, 3)

    def encode_tokens(self, tok_std):
        return self.ae.forward_encoder(tok_std, self._pos(tok_std.shape[0]))

    def encode(self, theta_aligned):
        return self.encode_tokens(self.tokens(theta_aligned))  # (B, 58, lat)

    def decode_tokens(self, z):
        return self.ae.forward_decoder(z, self._pos(z.shape[0]))

    def decode(self, z):
        """z (B,58,lat) -> aligned theta (B,4970)."""
        return untokenize(destandardize(self.decode_tokens(z), self.mu, self.sigma))

    def forward(self, tok_std):
        return self.ae(tok_std, self._pos(tok_std.shape[0]))  # z, y, zp

    def reconstruct(self, theta_aligned):
        return self.decode(self.encode(theta_aligned))


# ---------------------------------------------------------------- property heads
def act_onehot(act):
    return F.one_hot(act.long(), 2).float()


class ZBarHead(nn.Module):
    """Head on the mean token embedding z̄ (plus activation one-hot). kind 'linear' (ridge / linear probe)
    or 'mlp' (2 hidden layers). Output: 10 per-class recalls (sigmoid for mlp, raw for linear)."""

    def __init__(self, lat, kind="linear", hidden=256, use_act=True):
        super().__init__()
        self.kind, self.use_act = kind, use_act
        d_in = lat + (2 if use_act else 0)
        if kind == "linear":
            self.net = nn.Linear(d_in, 10)
        else:
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, 10))

    def forward(self, z, act):
        x = z.mean(dim=1)
        if self.use_act:
            x = torch.cat([x, act_onehot(act)], 1)
        out = self.net(x)
        return out if self.kind == "linear" else torch.sigmoid(out)


class TokenHead(nn.Module):
    """Per-class head: recall_c = f([z̄, z_dense_c, act]) with f shared across classes (the dense tokens
    are the per-class output rows)."""

    def __init__(self, lat, hidden=256, use_act=True):
        super().__init__()
        self.use_act = use_act
        d_in = 2 * lat + (2 if use_act else 0) + 10
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, z, act):
        B = z.shape[0]
        zbar = z.mean(dim=1, keepdim=True).expand(B, 10, z.shape[2])
        zd = z[:, DENSE_TOKENS]
        parts = [zbar, zd, torch.eye(10, device=z.device)[None].expand(B, 10, 10)]
        if self.use_act:
            parts.append(act_onehot(act)[:, None].expand(B, 10, 2))
        return torch.sigmoid(self.net(torch.cat(parts, 2))[..., 0])


class SanePsi(nn.Module):
    """ψ(θ, act) = head(encoder(tok(align(θ)))). `perms` (B,3,16) are fixed by the caller (set_perms) —
    e.g. computed at θ₀ and held along an edit path; if None they are recomputed from θ.detach()."""

    def __init__(self, sane: SaneZoo, head: nn.Module, theta_ref):
        super().__init__()
        self.sane, self.head = sane, head
        self.register_buffer("theta_ref", theta_ref.clone())
        self.perms = None
        self.restarts = 8

    def set_perms(self, P):
        self.perms = P

    def align(self, theta):
        P = self.perms
        if P is None:
            P, _ = weight_match_restarts(theta.detach(), self.theta_ref, restarts=self.restarts)
        elif P.shape[0] != theta.shape[0]:
            raise ValueError("fixed perms do not match the batch; call set_perms(None) or set_perms(P) for this batch")
        return permute_batched(theta, P), P

    def forward(self, theta, act):
        th, _ = self.align(theta)
        return self.head(self.sane.encode(th), act)

    def recon_error(self, theta):
        """Manifold distance: ||θ_al − decode(encode(θ_al))|| / ||θ_al|| per model (aligned space)."""
        th, _ = self.align(theta)
        rec = self.sane.reconstruct(th)
        return (rec - th).norm(dim=1) / th.norm(dim=1)


def save_sane(path, sane, heads, theta_ref, ref_idx, meta):
    torch.save({"cfg": sane.cfg, "mu": sane.mu.cpu(), "sigma": sane.sigma.cpu(), "ae": sane.ae.state_dict(),
                "heads": {k: (type(v).__name__, getattr(v, "kind", None), v.state_dict()) for k, v in heads.items()},
                "theta_ref": theta_ref.cpu(), "ref_idx": ref_idx, "meta": meta}, path)


def load_sane(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    sane = SaneZoo(ck["cfg"], ck["mu"], ck["sigma"]).to(device)
    sane.ae.load_state_dict(ck["ae"])
    lat = ck["cfg"]["ae:lat_dim"]
    heads = {}
    for k, (cls, kind, sd) in ck["heads"].items():
        h = TokenHead(lat) if cls == "TokenHead" else ZBarHead(lat, kind=kind)
        h.load_state_dict(sd)
        heads[k] = h.to(device)
    sane.eval()
    for h in heads.values():
        h.eval()
    return sane, heads, ck["theta_ref"].to(device), ck

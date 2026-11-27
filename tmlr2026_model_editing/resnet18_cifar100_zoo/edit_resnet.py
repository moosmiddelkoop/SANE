"""Algorithm-1 editing (weight space and SANE latent space) for the ResNet-18 zoo, with controls and data baselines.
All psi objects: psi(theta) -> (B, 100) predicted per-class recall for a batch of flat theta (B, P) on GPU."""
import math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import resnet_zoo as rz, sane_resnet as sr

NC = rz.NC
LAST = slice((rz.N_TOK // 512) * 512, rz.N_TOK)   # covering window of the fc tokens for window 512 (set by set_window)


def set_window(W):
    global LAST
    LAST = slice((rz.N_TOK // W) * W, rz.N_TOK)


def fc_param_indices():
    return rz.param_indices_of_tokens(rz.FC_TOKENS)


def last_param_indices():
    return rz.param_indices_of_tokens(LAST)


# ----------------------------------------------------------------- heads (same definitions as fit_heads.py)
class MLPHead(nn.Module):
    def __init__(s, d, h=512): super().__init__(); s.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, NC))
    def forward(s, x): return torch.sigmoid(s.net(x))


class TokHead(nn.Module):
    def __init__(s, d, h=512): super().__init__(); s.net = nn.Sequential(nn.Linear(d + NC, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(s, x):
        B = x.shape[0]; return torch.sigmoid(s.net(torch.cat([x, torch.eye(NC, device=x.device)[None].expand(B, -1, -1)], 2))[..., 0])


class SanePsi(nn.Module):
    """psi(theta) = head(z̄, z_fc) with z̄ = (S_other + sum z_last)/N_TOK. mode 'last': only the covering window of the fc tokens
    is re-encoded (S_other frozen at theta0 via set_batch); mode 'full': all windows re-encoded (expensive)."""

    def __init__(self, sane, heads_ck, zoo, head="token", mode="last"):
        super().__init__()
        self.sane, self.zoo, self.head_kind, self.mode = sane, zoo, head, mode
        lat = sane.lat
        if head == "token":
            self.head = TokHead(lat * (1 + rz.TOK_PER_CLASS)); self.head.load_state_dict(heads_ck["token"])
            self.fm = torch.as_tensor(heads_ck["fm"], dtype=torch.float32); self.fs = torch.as_tensor(heads_ck["fs"], dtype=torch.float32)
        else:
            self.head = MLPHead(lat); self.head.load_state_dict(heads_ck["mlp"])
            self.zm = torch.as_tensor(heads_ck["zm"], dtype=torch.float32); self.zs = torch.as_tensor(heads_ck["zs"], dtype=torch.float32)
        self.S_other = None
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()

    def to(self, dev):
        super().to(dev)
        for a in ("fm", "fs", "zm", "zs"):
            if hasattr(self, a): setattr(self, a, getattr(self, a).to(dev))
        return self

    @torch.no_grad()
    def set_batch(self, theta0):
        """freeze the contribution of all tokens outside the last window (mode 'last')."""
        tok = rz.tokenize(self.zoo.standardize(theta0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.sane.encode_model(tok)
        self.z0_full = z
        self.S_other = z[:, : LAST.start].sum(1)

    def z_last(self, theta):
        tok = rz.tokenize_slice(self.zoo.standardize(theta), LAST)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.sane.encode_windows(tok, self.sane.pos[LAST][None].expand(theta.shape[0], -1, -1))
        return z.float()

    def z_all(self, theta):
        """differentiable full encoding (mode 'full'), windows batched."""
        tok = rz.tokenize(self.zoo.standardize(theta)); B = theta.shape[0]; W = self.sane.W; outs = []
        for s in range(0, rz.N_TOK, W):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs.append(self.sane.encode_windows(tok[:, s:s + W], self.sane.pos[s:s + W][None].expand(B, -1, -1)).float())
        return torch.cat(outs, 1)

    def head_from_z(self, zbar, zfc):
        if self.head_kind == "token":
            x = torch.cat([zfc.reshape(zfc.shape[0], NC, -1), zbar[:, None].expand(-1, NC, -1)], 2)
            return self.head((x - self.fm) / self.fs)
        return self.head((zbar - self.zm) / self.zs)

    def forward(self, theta):
        if self.mode == "full":
            z = self.z_all(theta); zbar = z.mean(1); zfc = sr.zfc_from_z(z)
        else:
            zl = self.z_last(theta); zbar = (self.S_other + zl.sum(1)) / rz.N_TOK
            zfc = zl[:, rz.FC_TOKENS.start - LAST.start:].reshape(theta.shape[0], NC, rz.TOK_PER_CLASS, -1)
        return self.head_from_z(zbar, zfc)

    def score_z(self, zl):
        """psi from an edited last-window latent (SANE-Z)."""
        zbar = (self.S_other + zl.sum(1)) / rz.N_TOK
        zfc = zl[:, rz.FC_TOKENS.start - LAST.start:].reshape(zl.shape[0], NC, rz.TOK_PER_CLASS, -1)
        return self.head_from_z(zbar, zfc)

    def decode_last(self, zl, theta0):
        """decode a last-window latent and splice the covered parameters into theta0 -> theta (B,P)."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = self.sane.decode_windows(zl, self.sane.pos[LAST][None].expand(zl.shape[0], -1, -1)).float()
        y = y * self.sane.mask[LAST]
        return splice_tokens(theta0, y, LAST, self.zoo)

    @torch.no_grad()
    def recon_error_last(self, theta):
        """manifold distance of the last window: ||tok - dec(enc(tok))|| / ||tok|| in standardised token space."""
        tok = rz.tokenize_slice(self.zoo.standardize(theta), LAST)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = self.sane.encode_windows(tok, self.sane.pos[LAST][None].expand(theta.shape[0], -1, -1))
            y = self.sane.decode_windows(z, self.sane.pos[LAST][None].expand(theta.shape[0], -1, -1)).float() * self.sane.mask[LAST]
        return (y - tok).flatten(1).norm(dim=1) / tok.flatten(1).norm(dim=1)


def splice_tokens(theta0, tok_std, sl, zoo):
    """replace the parameters covered by token slice sl with the (standardised) token values tok_std (B, len, D_T)."""
    idx = rz.IDX[sl].to(theta0.device).reshape(-1); real = idx < rz.P
    vals = tok_std.reshape(tok_std.shape[0], -1)[:, real]
    pidx = idx[real]
    theta = theta0.clone()
    theta[:, pidx] = vals * zoo.sigma.to(theta0.device)[pidx] + zoo.mu.to(theta0.device)[pidx]
    return theta


class FcRidgePsi(nn.Module):
    """ridge on per-class fc-row statistics (+ class one-hot), differentiable in theta."""

    def __init__(self, ck):
        super().__init__()
        for k in ("W", "m", "s", "ym"): self.register_buffer(k, torch.as_tensor(ck[k], dtype=torch.float32))

    def forward(self, theta):
        f = rz.fc_row_statistics(theta)                                     # (B,100,7)
        x = torch.cat([f, torch.eye(NC, device=theta.device)[None].expand(theta.shape[0], -1, -1)], 2)
        return ((x - self.m) / self.s) @ self.W[:, 0] + self.ym if self.W.ndim == 2 else ((x - self.m) / self.s) @ self.W + self.ym


# ----------------------------------------------------------------- loss / stop / Algorithm 1
def loss_fn(pred, c, kind, beta=0.1, sign=+1):
    pc = pred.gather(1, c[:, None])[:, 0]
    if kind == "simple": return sign * pc
    mask = torch.ones_like(pred); mask.scatter_(1, c[:, None], 0.0)
    return sign * (pc - beta * (pred * mask).sum(1))


def edit(theta0, c, psi, task="unlearn", beta=0.1, eta=1e-3, max_steps=1000, pidx=None, direction=None,
         eval_fn=None, log_every=0, monitor=None, use_stop=True, oracle_stop=False):
    """oracle_stop: freeze an edit at the first log step where eval_fn (TRUE recall_c) meets the criterion (diagnostic ceiling, needs eval_fn and log_every)."""
    """Batched Algorithm 1 in weight space. pidx: LongTensor of editable parameter indices (gradient masked to them;
    step normalised by the norm of theta0 on those indices). eval_fn(theta)->(B,) true recall_c for trajectories."""
    kind = "boost" if task == "unlearn" else "simple"; sign = +1 if task == "unlearn" else -1
    B = theta0.shape[0]; dev = theta0.device
    theta = theta0.clone()
    with torch.no_grad(): pred0 = psi(theta0)
    p0 = pred0.gather(1, c[:, None])[:, 0]
    target = p0 * 0.4 if task == "unlearn" else torch.full_like(p0, 0.9)
    norm0 = (theta0[:, pidx] if pidx is not None else theta0).norm(dim=1, keepdim=True)
    active = torch.ones(B, dtype=torch.bool, device=dev); steps = torch.zeros(B, dtype=torch.long, device=dev)
    psi_stop_step = torch.full((B,), -1, dtype=torch.long, device=dev); oracle_step = torch.full((B,), -1, dtype=torch.long, device=dev)
    tp, tt, ts, tm = [], [], [], []
    for t in range(max_steps + 1):
        th = theta.detach().requires_grad_(True)
        pred = psi(th); pc = pred.gather(1, c[:, None])[:, 0].detach()
        done = (pc <= target) if task == "unlearn" else (pc >= target)
        if log_every and t % log_every == 0:
            tp.append(pc.clone()); ts.append(t)
            if eval_fn is not None: tt.append(eval_fn(theta.detach()))
            if monitor is not None: tm.append(monitor(theta.detach()))
            if oracle_stop:
                tr0 = tt[0]; done_true = (tt[-1] <= 0.4 * tr0) if task == "unlearn" else (tt[-1] >= 0.9)
                oracle_step = torch.where((oracle_step < 0) & done_true, torch.full_like(oracle_step, t), oracle_step)
                active = active & ~done_true
        psi_stop_step = torch.where((psi_stop_step < 0) & done, torch.full_like(psi_stop_step, t), psi_stop_step)
        if use_stop: active = active & ~done
        if t == max_steps or not active.any(): break
        if direction is None:
            g, = torch.autograd.grad(loss_fn(pred, c, kind, beta, sign).sum(), th)
        else:
            g = sign * direction
        with torch.no_grad():
            if pidx is not None:
                gm = torch.zeros_like(g); gm[:, pidx] = g[:, pidx]; g = gm
            g = g / (g.norm(dim=1, keepdim=True) + 1e-12) * norm0
            theta = theta - eta * g * active[:, None].float()
        steps += active.long()
    with torch.no_grad(): pred1 = psi(theta)
    out = dict(theta=theta.detach(), steps=steps, stopped=(~active if use_stop else psi_stop_step >= 0), pred0=pred0.detach(), pred1=pred1.detach(), target=target,
               psi_stop_step=psi_stop_step, oracle_step=oracle_step)
    if oracle_stop: out["stopped"] = oracle_step >= 0
    if tp:
        out["traj_pred"] = torch.stack(tp, 1); out["traj_step"] = np.array(ts)
        out["traj_true"] = torch.stack(tt, 1) if tt else None; out["traj_mon"] = torch.stack(tm, 1) if tm else None
    return out


def edit_z(theta0, c, psi: SanePsi, task="unlearn", beta=0.1, eta=1e-2, max_steps=1000, dense_only=True,
           eval_fn=None, log_every=0, gen=None):
    """SANE-Z on the last-window latent: z <- z - eta*||z0_mask||*g/||g||; theta_t = splice(decode(z_t)) into theta0.
    dense_only: edit only the fc-token latents. Returns also theta_identity (decode(z0) spliced) and a norm-matched random walk."""
    kind = "boost" if task == "unlearn" else "simple"; sign = +1 if task == "unlearn" else -1
    B = theta0.shape[0]; dev = theta0.device
    with torch.no_grad():
        z0 = psi.z_last(theta0); theta_id = psi.decode_last(z0, theta0); pred0 = psi.score_z(z0)
    p0 = pred0.gather(1, c[:, None])[:, 0]
    target = p0 * 0.4 if task == "unlearn" else torch.full_like(p0, 0.9)
    zmask = torch.ones(1, z0.shape[1], 1, device=dev)
    if dense_only: zmask[:] = 0; zmask[:, rz.FC_TOKENS.start - LAST.start:] = 1
    norm0 = (z0 * zmask).flatten(1).norm(dim=1).view(B, 1, 1)
    z = z0.clone(); active = torch.ones(B, dtype=torch.bool, device=dev); steps = torch.zeros(B, dtype=torch.long, device=dev)
    tp, tt, ts, tm = [], [], [], []
    for t in range(max_steps + 1):
        zz = z.detach().requires_grad_(True)
        pred = psi.score_z(zz); pc = pred.gather(1, c[:, None])[:, 0].detach()
        done = (pc <= target) if task == "unlearn" else (pc >= target)
        if log_every and t % log_every == 0:
            tp.append(pc.clone()); ts.append(t)
            with torch.no_grad():
                th_t = psi.decode_last(z.detach(), theta0)
                tm.append(psi.recon_error_last(th_t))
                if eval_fn is not None: tt.append(eval_fn(th_t))
        active = active & ~done
        if t == max_steps or not active.any(): break
        g, = torch.autograd.grad(loss_fn(pred, c, kind, beta, sign).sum(), zz)
        with torch.no_grad():
            g = g * zmask; g = g / (g.flatten(1).norm(dim=1).view(B, 1, 1) + 1e-12) * norm0
            z = z - eta * g * active.view(B, 1, 1).float()
        steps += active.long()
    with torch.no_grad():
        pred1 = psi.score_z(z); theta = psi.decode_last(z, theta0); dz = (z - z0).flatten(1).norm(dim=1)
        r = torch.randn(z0.shape, generator=gen).to(dev) * zmask
        r = r / (r.flatten(1).norm(dim=1).view(B, 1, 1) + 1e-12) * dz.view(B, 1, 1)
        theta_zr = psi.decode_last(z0 + r, theta0); pred_zr = psi.score_z(z0 + r)
    out = dict(theta=theta, steps=steps, stopped=~active, pred0=pred0, pred1=pred1, target=target, theta_identity=theta_id,
               theta_zrandom=theta_zr, pred_zrandom=pred_zr, z_delta_norm=dz, z0_norm=norm0.flatten())
    if tp:
        out["traj_pred"] = torch.stack(tp, 1); out["traj_step"] = np.array(ts)
        out["traj_true"] = torch.stack(tt, 1) if tt else None; out["traj_mon"] = torch.stack(tm, 1)
    return out


# ----------------------------------------------------------------- weight-space controls
def random_matched(theta0, delta, pidx, gen=None):
    v = torch.randn(delta.shape[0], len(pidx), generator=gen).to(delta.device)
    tau = delta[:, pidx].norm(dim=1, keepdim=True)
    out = theta0.clone(); out[:, pidx] += v / (v.norm(dim=1, keepdim=True) + 1e-12) * tau
    return out


def random_full_matched(theta0, delta, gen=None):
    """random direction over ALL parameters, norm-matched to the full edit norm (control for full-gradient edits)."""
    v = torch.randn(delta.shape, generator=gen).to(delta.device)
    return theta0 + v / (v.norm(dim=1, keepdim=True) + 1e-12) * delta.norm(dim=1, keepdim=True)


def bias_matched(theta0, delta, c, task="unlearn"):
    tau = delta.norm(dim=1); out = theta0.clone(); a0 = rz.OFFSETS["fc.bias"][0]
    out[torch.arange(len(c), device=c.device), a0 + c] += (-1 if task == "unlearn" else +1) * tau
    return out


def rowscale_matched(theta0, delta, c, task="unlearn"):
    """shrink (unlearn) / grow (uplearn) output row W_c along itself by the edit norm (shrink clipped at zero)."""
    tau = delta.norm(dim=1); out = theta0.clone(); a0, _, _ = rz.OFFSETS["fc.weight"]
    ar = torch.arange(len(c), device=c.device)
    rows = torch.stack([out[i, a0 + ci * 512: a0 + (ci + 1) * 512] for i, ci in zip(ar.tolist(), c.tolist())])
    n = rows.norm(dim=1, keepdim=True)
    fac = (1 - tau[:, None] / (n + 1e-12)).clamp(min=0) if task == "unlearn" else (1 + tau[:, None] / (n + 1e-12))
    for i, ci in zip(ar.tolist(), c.tolist()):
        out[i, a0 + ci * 512: a0 + (ci + 1) * 512] = rows[i] * fac[i]
    return out


def delta_shares(delta, c):
    """squared-norm shares of delta: fc row c, other fc rows, bias c, other biases, conv (rest)."""
    a0, _, _ = rz.OFFSETS["fc.weight"]; b0, _, _ = rz.OFFSETS["fc.bias"]
    tot = (delta ** 2).sum(1) + 1e-30; B = delta.shape[0]; ar = torch.arange(B, device=delta.device)
    W = delta[:, a0:a0 + NC * 512].reshape(B, NC, 512); bvec = delta[:, b0:b0 + NC]
    wc = (W[ar, c] ** 2).sum(1); wall = (W ** 2).sum((1, 2)); bc = bvec[ar, c] ** 2; ball = (bvec ** 2).sum(1)
    conv = tot - wall - ball
    return dict(share_fc_w_c=wc / tot, share_fc_w_other=(wall - wc) / tot, share_fc_b_c=bc / tot, share_fc_b_other=(ball - bc) / tot,
                share_conv=conv / tot, sign_b_c=torch.sign(delta[ar, b0 + c]), delta_norm=tot.sqrt(), rel_change_fc_row=None)


# ----------------------------------------------------------------- data baselines (per-model Adam, BN in eval mode)
def finetune(theta0, bufs, X, y, lr, steps, sign=+1, batch=64, gen=None, pidx=None, stop_fn=None, mp=False):
    """X (B,n,3,32,32), y (B,n). sign +1 = descend CE (learn), -1 = ascend (forget). Returns theta, steps_used, stopped."""
    B, n = y.shape; dev = theta0.device
    theta = theta0.clone()
    m1 = torch.zeros_like(theta); m2 = torch.zeros_like(theta)
    active = torch.ones(B, dtype=torch.bool, device=dev); used = torch.zeros(B, dtype=torch.long, device=dev)
    lr = torch.as_tensor(lr, dtype=torch.float32, device=dev).view(-1, 1).expand(B, 1)
    for t in range(1, steps + 1):
        if stop_fn is not None:
            with torch.no_grad(): active = active & ~stop_fn(theta)
            if not active.any(): break
        g = torch.zeros_like(theta)
        for i in torch.nonzero(active).flatten().tolist():
            idx = torch.randint(0, n, (batch,), generator=gen).to(dev) if batch < n else torch.arange(n, device=dev)
            th = theta[i].detach().requires_grad_(True)
            loss = F.cross_entropy(rz.resnet18_forward(th, bufs[i], X[i, idx], maxpool=mp), y[i, idx])
            g[i], = torch.autograd.grad(loss, th)
        g = sign * g
        if pidx is not None:
            gm = torch.zeros_like(g); gm[:, pidx] = g[:, pidx]; g = gm
        with torch.no_grad():
            m1 = 0.9 * m1 + 0.1 * g; m2 = 0.999 * m2 + 0.001 * g * g
            theta -= lr * (m1 / (1 - 0.9 ** t)) / ((m2 / (1 - 0.999 ** t)).sqrt() + 1e-8) * active[:, None].float()
        used += active.long()
    if stop_fn is not None:
        with torch.no_grad(): active = active & ~stop_fn(theta)
    return theta.detach(), used, ~active


# ----------------------------------------------------------------- scoring
def scores(rec0, rec1, c, task="unlearn"):
    sign = 1.0 if task == "unlearn" else -1.0
    move = sign * (rec0 - rec1); ar = torch.arange(len(c), device=c.device)
    mask = torch.ones_like(move, dtype=torch.bool); mask[ar, c] = False
    ctrl0 = (rec0 * mask).sum(1) / (NC - 1); ctrl1 = (rec1 * mask).sum(1) / (NC - 1)
    worst = (rec0 - rec1).masked_fill(~mask, -1e9).max(1).values            # largest control recall DROP
    n_dmg = ((rec0 - rec1) > 0.1).masked_fill(~mask, False).sum(1)
    return dict(target_move=move[ar, c], Dt0=rec0[ar, c], Dt1=rec1[ar, c], Dr0=ctrl0, Dr1=ctrl1, ctrl_change=ctrl1 - ctrl0,
                worst_ctrl_drop=worst, n_ctrl_damaged=n_dmg)


def criterion_met(rec1_c, rec0_c, task, tol=0.0):
    return rec1_c <= 0.4 * rec0_c + tol if task == "unlearn" else rec1_c >= 0.9 - tol


# ----------------------------------------------------------------- per-layer norm-matched random control (= edit_cnn.random_matched on the CNN zoos)
def _layer_groups():
    """21 contiguous parameter groups in state-dict order: each of the 20 conv layers pooled with its BatchNorm weight and bias, and fc.weight with fc.bias."""
    g = [(cw, rz.OFFSETS[cw + ".weight"][0], rz.OFFSETS[bn + ".bias"][1]) for cw, bn, _ in rz.CONV_SPECS]
    g.append(("fc", rz.OFFSETS["fc.weight"][0], rz.OFFSETS["fc.bias"][1]))
    assert g[0][1] == 0 and g[-1][2] == rz.P and all(g[i][2] == g[i + 1][1] for i in range(len(g) - 1))
    return g
LAYER_GROUPS = _layer_groups()
LAYER_GROUP_NAMES = [n for n, _, _ in LAYER_GROUPS]


def layer_group_norms(delta):
    """(B, P) -> (B, 21) L2 norm of delta per layer group."""
    return torch.stack([delta[:, a:b].norm(dim=1) for _, a, b in LAYER_GROUPS], 1)


def random_layer_matched(theta0, delta, gen=None):
    """per-layer norm-matched random direction: for each group g, v_g ~ N(0, I) drawn on the CPU generator gen, scaled to ||delta_g||."""
    v = torch.randn(delta.shape, generator=gen).to(delta.device); out = theta0.clone()
    for _, a, b in LAYER_GROUPS:
        tau = delta[:, a:b].norm(dim=1, keepdim=True); vg = v[:, a:b]
        out[:, a:b] += vg / (vg.norm(dim=1, keepdim=True) + 1e-12) * tau
    return out

"""Algorithm 1 (weight-space editing through a metanet) and baselines for the Small CNN Zoo."""
import os, math, time, json
os.environ.setdefault("KMP_AFFINITY", "disabled")
import numpy as np
import torch
import torch.nn.functional as F
import cnnzoo, meta_cnn as mc
from cnnzoo import SEG

K = 10


# ----------------------------------------------------------------- metanet loading
def load_metanet(path):
    dev = cnnzoo.DEV
    head = None
    if "@" in path:
        path, head = path.split("@")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "ae" in ck and "heads" in ck:  # SANE checkpoint written by run_sane_pretrain.py
        import sane_zoo as sz
        sane, heads, theta_ref, _ = sz.load_sane(path, device=dev)
        head = head or "token"
        net = sz.SanePsi(sane, heads[head], theta_ref).to(dev); net.eval()
        net.aligned = not ck.get("meta", {}).get("args", {}).get("no_align", False)
        for p in net.parameters():
            p.requires_grad_(False)
        net.spec = f"sane_{head}"
        return net
    spec = ck["spec"]; sd = ck["state_dict"]
    if spec in ("mlp_paper",):
        net = mc.MLPMeta(use_act=False)
    elif spec in ("mlp_act",):
        net = mc.MLPMeta(use_act=True)
    elif spec in ("mlp_aug", "mlp_std"):
        net = mc.MLPMeta(use_act=True, mu=sd["mu"], sd=sd["sd"])
    elif spec.startswith("graph"):
        net = mc.ChannelGraphNet(d=64, rounds=int(spec[5:]))
    elif spec.startswith("probe_"):
        _, mode, P = spec.split("_"); P = int(P)
        init = None
        if mode in ("real", "fixed"):
            init = torch.tanh(sd["u"]).numpy()
        net = mc.ProbeNet(P=P, mode=mode, init_images=init)
    else:
        raise ValueError(spec)
    net.load_state_dict(sd); net.to(dev); net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def psi_fn(net):
    return lambda theta, act: net(theta, act)


# ----------------------------------------------------------------- loss & Algorithm 1
def loss_fn(pred, c, kind, beta=0.1, sign=+1):
    pc = pred.gather(1, c[:, None])[:, 0]
    if kind == "simple":
        f = pc
    else:
        mask = torch.ones_like(pred); mask.scatter_(1, c[:, None], 0.0)
        f = pc - beta * (pred * mask).sum(1)
    return sign * f


def true_recall(theta, act, Xte, yte, chunk=None):
    rec, acc = cnnzoo.per_class_recall(theta, act, Xte, yte, chunk=chunk)
    return rec


def edit(theta0, act, c, psi, task="unlearn", kind=None, beta=0.1, eta=1e-3, step_mode="rel",
         stop=None, thresh=None, max_steps=2000, grad_mask=None, direction=None,
         Xte=None, yte=None, log_every=0, monitor=None):
    """Batched Algorithm 1.
    theta0 (B,N); act (B,); c (B,) long; psi(theta, act)->(B,K).
    task: 'unlearn' (Boost loss, relative stop 60% drop) or 'uplearn' (Simple loss, absolute stop 0.9) by default.
    step_mode 'rel': theta <- theta - eta*||theta0||*g/||g|| ; 'raw': theta <- theta - eta*g (paper).
    grad_mask (N,): multiplicative mask on g. direction (B,N): fixed direction replacing g.
    Returns dict(theta, steps, stopped, pred0, pred_final, traj_*)."""
    if kind is None:
        kind = "boost" if task == "unlearn" else "simple"
    if stop is None:
        stop = "relative" if task == "unlearn" else "absolute"
    if thresh is None:
        thresh = 0.6 if task == "unlearn" else 0.9
    B, N = theta0.shape
    sign = +1 if task == "unlearn" else -1
    theta = theta0.clone()
    with torch.no_grad():
        pred0 = psi(theta0, act)
    p0 = pred0.gather(1, c[:, None])[:, 0]
    if stop == "relative":
        target = p0 * (1 - thresh) if task == "unlearn" else p0 * (1 + thresh)
    else:
        target = torch.full_like(p0, float(thresh))
    norm0 = theta0.norm(dim=1, keepdim=True)
    active = torch.ones(B, dtype=torch.bool, device=theta0.device)
    steps = torch.zeros(B, dtype=torch.long, device=theta0.device)
    traj_pred, traj_true, traj_step, traj_mon = [], [], [], []
    pred = None
    for t in range(max_steps + 1):
        th = theta.detach().requires_grad_(True)
        pred = psi(th, act)
        pc = pred.gather(1, c[:, None])[:, 0].detach()
        done = (pc <= target) if task == "unlearn" else (pc >= target)
        if log_every and t % log_every == 0:
            traj_pred.append(pc.clone()); traj_step.append(t)
            if Xte is not None:
                traj_true.append(true_recall(theta.detach(), act, Xte, yte).gather(1, c[:, None])[:, 0])
            if monitor is not None:
                with torch.no_grad():
                    traj_mon.append(monitor(theta.detach()))
        active = active & ~done
        if t == max_steps or not active.any():
            break
        if direction is None:
            f = loss_fn(pred, c, kind, beta, sign).sum()
            g, = torch.autograd.grad(f, th)
        else:
            g = sign * direction
        if grad_mask is not None:
            g = g * grad_mask[None, :]
        with torch.no_grad():
            if step_mode == "rel":
                g = g / (g.norm(dim=1, keepdim=True) + 1e-12) * norm0
            theta = theta - eta * g * active[:, None].float()
        steps += active.long()
    with torch.no_grad():
        pred_final = psi(theta, act)
    out = dict(theta=theta.detach(), steps=steps, stopped=~active, pred0=pred0.detach(), pred_final=pred_final.detach(),
               target=target.detach())
    if traj_pred:
        out["traj_pred"] = torch.stack(traj_pred, 1); out["traj_step"] = np.array(traj_step)
        out["traj_true"] = torch.stack(traj_true, 1) if traj_true else None
        out["traj_mon"] = torch.stack(traj_mon, 1) if traj_mon else None
    return out


def edit_z(theta0, act, c, net, task="unlearn", kind=None, beta=0.1, eta=1e-2, thresh=None, max_steps=2000,
           dense_only=False, Xte=None, yte=None, log_every=0, zrandom=True, gen=None, psi_theta=None, monitor=None):
    """SANE-Z: Algorithm 1 in SANE's token-latent space. net: sane_zoo.SanePsi with fixed perms for this batch.
    z <- z - eta*||z0||*g/||g|| on f(head(z)); same Boost/Simple loss and stop rule as edit(); theta_t = decode(z_t)
    (de-standardised, un-permuted). Also returns theta_identity = decode(z0) and, if zrandom, a norm-matched
    random walk in z decoded the same way.
    psi_theta(theta, act): if given ("zdec"), the loss and stop rule are evaluated on psi_theta(decode(z)) with the gradient
    taken through the decoder, i.e. the edit is constrained to the decoder's image and scored by a weight-space ψ
    (SANE round trip or an external metanet such as the graph net)."""
    import sane_zoo as sz
    if kind is None:
        kind = "boost" if task == "unlearn" else "simple"
    if thresh is None:
        thresh = 0.6 if task == "unlearn" else 0.9
    B = theta0.shape[0]
    sign = +1 if task == "unlearn" else -1
    sane, head = net.sane, net.head
    P0 = net.perms
    Pinv = sz.invert_perms(P0)
    decode = lambda z: sz.permute_batched(sane.decode(z), Pinv)
    score = (lambda zz: head(zz, act)) if psi_theta is None else (lambda zz: psi_theta(decode(zz), act))
    with torch.no_grad():
        th_al = sz.permute_batched(theta0, P0)
        z0 = sane.encode(th_al)
        theta_id = decode(z0)
        pred0 = score(z0)
    p0 = pred0.gather(1, c[:, None])[:, 0]
    target = p0 * (1 - thresh) if task == "unlearn" else torch.full_like(p0, float(thresh))
    zmask = torch.ones(1, sz.N_TOK, 1, device=z0.device)
    if dense_only:
        zmask = torch.zeros_like(zmask); zmask[:, sz.DENSE_TOKENS] = 1.0
    norm0 = z0.flatten(1).norm(dim=1).view(B, 1, 1)
    z = z0.clone()
    active = torch.ones(B, dtype=torch.bool, device=z.device); steps = torch.zeros(B, dtype=torch.long, device=z.device)
    traj_pred, traj_true, traj_step, traj_mon = [], [], [], []
    for t in range(max_steps + 1):
        zz = z.detach().requires_grad_(True)
        pred = score(zz)
        pc = pred.gather(1, c[:, None])[:, 0].detach()
        done = (pc <= target) if task == "unlearn" else (pc >= target)
        if log_every and t % log_every == 0:
            traj_pred.append(pc.clone()); traj_step.append(t)
            with torch.no_grad():
                th_t = decode(z.detach())
                traj_mon.append((monitor or net.recon_error)(th_t))
                if Xte is not None:
                    traj_true.append(true_recall(th_t, act, Xte, yte).gather(1, c[:, None])[:, 0])
        active = active & ~done
        if t == max_steps or not active.any():
            break
        f = loss_fn(pred, c, kind, beta, sign).sum()
        g, = torch.autograd.grad(f, zz)
        g = g * zmask
        with torch.no_grad():
            g = g / (g.flatten(1).norm(dim=1).view(B, 1, 1) + 1e-12) * norm0
            z = z - eta * g * active.view(B, 1, 1).float()
        steps += active.long()
    with torch.no_grad():
        pred_final = score(z)
        theta = decode(z)
        dz = (z - z0)
    out = dict(theta=theta.detach(), steps=steps, stopped=~active, pred0=pred0.detach(), pred_final=pred_final.detach(),
               target=target.detach(), theta_identity=theta_id, z_delta_norm=dz.flatten(1).norm(dim=1),
               z0_norm=norm0.flatten(), pred_identity=pred0.detach())
    if traj_pred:
        out["traj_pred"] = torch.stack(traj_pred, 1); out["traj_step"] = np.array(traj_step)
        out["traj_true"] = torch.stack(traj_true, 1) if traj_true else None
        out["traj_mon"] = torch.stack(traj_mon, 1)
    if zrandom:
        with torch.no_grad():
            r = torch.randn(z0.shape, generator=gen, device="cpu").to(z0.device) * zmask
            r = r / (r.flatten(1).norm(dim=1).view(B, 1, 1) + 1e-12) * dz.flatten(1).norm(dim=1).view(B, 1, 1)
            zr = z0 + r
            out["theta_zrandom"] = decode(zr); out["pred_zrandom"] = score(zr)
    return out


# ----------------------------------------------------------------- baselines
LAYER_GROUPS = [("c1w", "c1b"), ("c2w", "c2b"), ("c3w", "c3b"), ("dw", "db")]


def group_idx(w, b):
    return torch.cat([torch.arange(SEG[w][0], SEG[w][1]), torch.arange(SEG[b][0], SEG[b][1])])


def per_layer_norms(delta):
    return {w: delta[:, group_idx(w, b)].norm(dim=1) for (w, b) in LAYER_GROUPS}


def random_matched(theta0, delta, gen=None):
    """Random vector with the same per-layer norms as delta."""
    out = torch.zeros_like(delta)
    for (w, b) in LAYER_GROUPS:
        idx = group_idx(w, b)
        tau = delta[:, idx].norm(dim=1, keepdim=True)
        v = torch.randn(delta.shape[0], len(idx), generator=gen).to(delta.device)
        out[:, idx] = v / (v.norm(dim=1, keepdim=True) + 1e-12) * tau
    return theta0 + out


def bias_only_matched(theta0, delta, c, task="unlearn"):
    """Shift only output bias b_c by the total edit norm."""
    tau = delta.norm(dim=1)
    out = theta0.clone()
    out[torch.arange(len(c), device=c.device), SEG["db"][0] + c] += (-1 if task == "unlearn" else +1) * tau
    return out


def bias_direction(c, N=cnnzoo.D):
    d = torch.zeros(len(c), N, device=c.device)
    d[torch.arange(len(c), device=c.device), SEG["db"][0] + c] = 1.0
    return d


def output_layer_mask():
    m = torch.zeros(cnnzoo.D, device=cnnzoo.DEV)
    m[SEG["dw"][0]:SEG["dw"][1]] = 1; m[SEG["db"][0]:SEG["db"][1]] = 1
    return m


def finetune(theta0, act, X, y, lr, steps, sign=+1, batch=64, gen=None, mask=None, stop_fn=None):
    """Per-model Adam on CE (sign=+1 descend = learn; sign=-1 ascend = unlearn). X (B,n,1,28,28), y (B,n).
    stop_fn(theta)->bool (B,) freezes models that meet the criterion (checked every step). Returns (theta, steps_used, stopped)."""
    B, n = y.shape
    theta = theta0.clone().requires_grad_(True)
    m1 = torch.zeros_like(theta); m2 = torch.zeros_like(theta)
    dev = theta0.device
    lr = torch.as_tensor(lr).float().view(-1, 1).expand(B, 1).to(dev)
    ar = torch.arange(B, device=dev)[:, None]
    active = torch.ones(B, dtype=torch.bool, device=dev); used = torch.zeros(B, dtype=torch.long, device=dev)
    for t in range(1, steps + 1):
        if stop_fn is not None:
            with torch.no_grad():
                active = active & ~stop_fn(theta.detach())
            if not active.any():
                break
        idx = torch.arange(n, device=dev)[None].expand(B, n) if batch >= n else torch.randint(0, n, (B, batch), generator=gen).to(dev)
        logits = cnnzoo.forward(theta, X[ar, idx], act, shared_x=False)          # (B,b,K)
        loss = F.cross_entropy(logits.reshape(-1, K), y[ar, idx].reshape(-1), reduction="sum") / B
        g, = torch.autograd.grad(loss, theta)
        g = sign * g
        if mask is not None:
            g = g * mask
        with torch.no_grad():
            m1 = 0.9 * m1 + 0.1 * g; m2 = 0.999 * m2 + 0.001 * g * g
            theta -= lr * (m1 / (1 - 0.9 ** t)) / ((m2 / (1 - 0.999 ** t)).sqrt() + 1e-8) * active[:, None].float()
        used += active.long()
    if stop_fn is not None:
        with torch.no_grad():
            active = active & ~stop_fn(theta.detach())
    return theta.detach(), used, ~active


def probe_pseudo_data(net, theta0, act, c, n_sel=None):
    """Pick the probes the model (currently) assigns the highest class-c probability to. Returns X (B,n,1,28,28), y (B,n)=c."""
    with torch.no_grad():
        probes = net.probes()                                 # (P,1,28,28)
        z = cnnzoo.forward(theta0, probes, act)               # (B,P,K)
        pc = torch.softmax(z, -1).gather(2, c[:, None, None].expand(-1, z.shape[1], 1))[..., 0]  # (B,P)
    n_sel = n_sel or max(4, probes.shape[0] // 8)
    top = pc.topk(n_sel, dim=1).indices                       # (B,n)
    X = probes[top]                                           # (B,n,1,28,28)
    y = c[:, None].expand(-1, n_sel).clone()
    return X, y


# ----------------------------------------------------------------- scoring
def scores(rec0, rec1, c, task="unlearn"):
    sign = 1.0 if task == "unlearn" else -1.0
    move = sign * (rec0 - rec1)                               # positive = moved in desired direction
    tgt = move.gather(1, c[:, None])[:, 0]
    mask = torch.ones_like(move, dtype=torch.bool); mask.scatter_(1, c[:, None], False)
    ctrl0 = (rec0 * mask).sum(1) / (K - 1); ctrl1 = (rec1 * mask).sum(1) / (K - 1)
    worst_ctrl = (sign * (rec0 - rec1)).masked_fill(~mask, -1e9).max(1).values  # largest collateral move in the edit direction
    return dict(target_move=tgt, Dt0=rec0.gather(1, c[:, None])[:, 0], Dt1=rec1.gather(1, c[:, None])[:, 0],
                Dr0=ctrl0, Dr1=ctrl1, ctrl_change=ctrl1 - ctrl0, worst_ctrl_move=worst_ctrl)


def criterion_met(rec1_c, rec0_c, task, tol=0.0):
    """True-recall version of the stop criterion (relative 60% drop / absolute 0.9)."""
    if task == "unlearn":
        return rec1_c <= 0.4 * rec0_c + tol
    return rec1_c >= 0.9 - tol

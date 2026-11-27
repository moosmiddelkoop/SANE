"""Unified blobs unlearning protocol (CNN-zoo protocol on the 2-16-16-6 toy zoo)."""
import os, numpy as np, torch
import toyzoo as T
from toyzoo import K, per_class_recall, forward_flat, layer_slices, unflatten
from exp import MLPMeta, NeuronNet
HERE = os.path.dirname(os.path.abspath(__file__))
SL = layer_slices()
ZOO = os.environ.get("ZOO", os.path.join(HERE, "zoo_50k.npz"))
ETA, BETA, REL, MAXS, LOG = 1e-3, 0.1, 0.4, 2000, 20
TOL, CTRL_TOL = 0.05, 0.05
FIXED = 100

def load_zoo():
    Z = np.load(ZOO); return {k: Z[k] for k in Z.files}

def tensor_scales(z):
    tr = z["split"] == "train"
    X = np.concatenate([z["theta"][c][tr] for c in range(3)])[:20000]
    return [float(p.std()) for p in unflatten(torch.tensor(X, dtype=torch.float32))]

def load_net(spec, path, z):
    if spec == "mlp_raw":
        net = MLPMeta(422)
    elif spec == "neuron_r1":
        net = NeuronNet(64, 1, wscale=tensor_scales(z))
    else:
        raise ValueError(spec)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net

NETS = {  # editor / judge files (seed 0 = editor, seed 1 = judge)
    ("mlp_raw", 0): "runs_new/mlp_raw_s0_f1.0.pt", ("mlp_raw", 1): "runs_new/mlp_raw_s1_f1.0.pt",
    ("neuron_r1", 0): "rescored/runs/neuron_r1_s0_f1.0.pt", ("neuron_r1", 1): "runs_new/neuron_r1_s1_f1.0.pt"}

def eval_set():   # same set as the stored zoo recall
    return T.make_test_set()

def probe_set():  # held-out set for the oracle stop (independent draw)
    return T.make_test_set(n_per_class=300, seed=1)

def edit_set(z, n_per_class=200, seed=0):
    te = np.where(z["split"] == "test")[0]
    R = z["recall"][-1, te]
    rng = np.random.default_rng(seed); M, C = [], []
    for c in range(K):
        ok = np.where((R > 0).all(1) & (R[:, c] >= 0.5))[0]
        pick = np.sort(rng.choice(ok, n_per_class, replace=False))
        M.append(te[pick]); C.append(np.full(n_per_class, c))
    return np.concatenate(M), np.concatenate(C)

def out_mask():
    m = torch.zeros(T.N_PARAMS); m[SL["W2"]] = 1; m[SL["b2"]] = 1; return m

def walk(theta0, c, psi, mask=None, Xp=None, yp=None, beta=BETA, eta=ETA, max_steps=MAXS, log_every=LOG):
    """Full-length normalised-step Algorithm-1 walk (never terminated early) on the Boost loss
    f = psi_c - beta*sum_{k!=c} psi_k. Records psi-stop state (first step with psi_c <= REL*psi_c(theta0),
    checked before each step), oracle state (first logged step with true probe recall_c <= REL*recall_c(theta0)),
    fixed-step state, theta snapshots and psi/true recall at every logged step."""
    B = theta0.shape[0]; ar = torch.arange(B)
    theta = theta0.clone(); n0 = theta0.norm(dim=1)
    with torch.no_grad():
        p0 = psi(theta0)[ar, c]
    r0p, _ = per_class_recall(theta0, Xp, yp); t0 = r0p[ar, c]
    oth = torch.ones(B, K); oth[ar, c] = 0
    ms_fired = torch.zeros(B, dtype=torch.bool); ms_step = torch.full((B,), max_steps); th_ms = theta0.clone()
    or_met = torch.zeros(B, dtype=torch.bool); or_step = torch.full((B,), -1); th_or = theta0.clone()
    best = t0.clone(); th_best = theta0.clone(); best_step = torch.zeros(B, dtype=torch.long)
    nlog = max_steps // log_every + 1
    snaps = torch.zeros(nlog, B, theta0.shape[1]); tp = torch.zeros(B, nlog); tt = torch.zeros(B, nlog); tpk = torch.zeros(B, nlog, K)
    th_fix = None
    for t in range(max_steps + 1):
        th = theta.detach().requires_grad_(t < max_steps)
        with torch.set_grad_enabled(t < max_steps):
            P = psi(th)
        pc = P[ar, c].detach()
        newly = (~ms_fired) & (pc <= REL * p0)
        ms_fired |= newly; ms_step[newly] = t; th_ms[newly] = theta[newly]
        if t % log_every == 0:
            k = t // log_every
            rp, _ = per_class_recall(theta, Xp, yp); tc = rp[ar, c]
            snaps[k] = theta; tp[:, k] = pc; tt[:, k] = tc; tpk[:, k] = P.detach()
            hit = (~or_met) & (tc <= REL * t0)
            or_met |= hit; or_step[hit] = t; th_or[hit] = theta[hit]
            imp = tc < best; best[imp] = tc[imp]; th_best[imp] = theta[imp]; best_step[imp] = t
        if t == FIXED:
            th_fix = theta.clone()
        if t == max_steps:
            break
        f = (P[ar, c] - beta * (P * oth).sum(1)).sum()
        g, = torch.autograd.grad(f, th)
        if mask is not None:
            g = g * mask[None]
        gn = g.norm(dim=1, keepdim=True)
        step = torch.where(gn > 0, g / gn.clamp_min(1e-12), torch.zeros_like(g)) * (eta * n0)[:, None]
        theta = (theta - step).detach()
    th_ms[~ms_fired] = theta[~ms_fired]
    th_or_final = torch.where(or_met[:, None], th_or, th_best)
    or_step_final = torch.where(or_met, or_step, best_step)
    return dict(ms=th_ms, ms_step=ms_step, stopped=ms_fired, p0=p0, probe_t0=t0,
                oracle=th_or_final, oracle_step=or_step_final, oracle_met=or_met, fixed=th_fix, final=theta,
                snaps=snaps, traj_psi=tp, traj_true=tt, traj_psi_all=tpk)

def adam_ft(theta0, X, y, lr, steps, sign=+1, batch=64, gen=None, stop_fn=None):
    """Batched Adam fine-tuning; sign=-1 = gradient ascent. stop_fn(theta)->bool mask checked before each step
    (stopped models are frozen). Returns theta, steps taken, stopped flag."""
    B, n = y.shape; ar = torch.arange(B)[:, None]
    theta = theta0.clone(); m1 = torch.zeros_like(theta); m2 = torch.zeros_like(theta)
    lr = lr if torch.is_tensor(lr) else torch.full((B,), float(lr))
    stopped = torch.zeros(B, dtype=torch.bool); nsteps = torch.full((B,), steps)
    for t in range(steps + 1):
        if stop_fn is not None:
            s = (~stopped) & stop_fn(theta); stopped |= s; nsteps[s] = t
            if bool(stopped.all()): break
        if t == steps: break
        idx = torch.randint(0, n, (B, batch), generator=gen)
        th = theta.clone().requires_grad_(True)
        lo = forward_flat(th, X[ar, idx])
        L = torch.nn.functional.cross_entropy(lo.reshape(-1, K), y[ar, idx].reshape(-1), reduction="none")
        g, = torch.autograd.grad(sign * L.reshape(B, batch).mean(1).sum(), th)
        with torch.no_grad():
            m1.mul_(0.9).add_(g, alpha=0.1); m2.mul_(0.999).addcmul_(g, g, value=0.001)
            upd = lr[:, None] * (m1 / (1 - 0.9 ** (t + 1))) / ((m2 / (1 - 0.999 ** (t + 1))).sqrt() + 1e-8)
            theta = theta - upd * (~stopped)[:, None].float()
    return theta, nsteps, stopped

def class_data(rng, c, n, exclude=False):
    B = len(c); labs = np.stack([rng.choice([k for k in range(K) if k != ci], size=n) if exclude else np.full(n, ci) for ci in c])
    X = np.stack([T.sample_points(rng, l) for l in labs])
    return torch.tensor(X), torch.tensor(labs)

def relearn(theta1, c, seed=123, ks=(1, 5, 20), lr=1e-2, n_forget=32, n_retain=160, Xe=None, ye=None):
    """Relearning attack: full-batch Adam (lr 1e-2) on a fixed mixed batch of 32 forget + 160 retain samples
    (true labels); same data per model across methods (fixed seed). Returns {k: recall (B,K)}."""
    rng = np.random.default_rng(seed); B = len(c)
    labs = np.stack([np.concatenate([np.full(n_forget, ci), rng.choice([k for k in range(K) if k != ci], size=n_retain)]) for ci in c])
    X = torch.tensor(np.stack([T.sample_points(rng, l) for l in labs])); y = torch.tensor(labs)
    theta = theta1.clone(); m1 = torch.zeros_like(theta); m2 = torch.zeros_like(theta); out = {}
    out[0] = per_class_recall(theta, Xe, ye)[0]
    for t in range(1, max(ks) + 1):
        th = theta.clone().requires_grad_(True)
        L = torch.nn.functional.cross_entropy(forward_flat(th, X).reshape(-1, K), y.reshape(-1), reduction="none").reshape(B, -1).mean(1).sum()
        g, = torch.autograd.grad(L, th)
        with torch.no_grad():
            m1.mul_(0.9).add_(g, alpha=0.1); m2.mul_(0.999).addcmul_(g, g, value=0.001)
            theta = theta - lr * (m1 / (1 - 0.9 ** t)) / ((m2 / (1 - 0.999 ** t)).sqrt() + 1e-8)
        if t in ks:
            out[t] = per_class_recall(theta, Xe, ye)[0]
    return out

BLOCKS = ["W1", "b1", "W2", "b2", "W3_c", "W3_other", "b3_c", "b3_other"]
def block_shares(d, c):
    """d (B,N) numpy edit; toyzoo W0/b0/W1/b1/W2/b2 renamed W1/b1/W2/b2/W3/b3 (paper numbering).
    W3 is (16,K) row-major: output unit c's incoming weights are W3[:, c]."""
    B = len(c); ar = np.arange(B); tot = (d ** 2).sum(1) + 1e-30
    W3 = d[:, SL["W2"]].reshape(B, 16, K) ** 2; b3 = d[:, SL["b2"]] ** 2
    w3c = W3[ar, :, c].sum(1); b3c = b3[ar, c]
    s = {"W1": (d[:, SL["W0"]] ** 2).sum(1), "b1": (d[:, SL["b0"]] ** 2).sum(1), "W2": (d[:, SL["W1"]] ** 2).sum(1),
         "b2": (d[:, SL["b1"]] ** 2).sum(1), "W3_c": w3c, "W3_other": W3.sum((1, 2)) - w3c, "b3_c": b3c, "b3_other": b3.sum(1) - b3c}
    return {k: v / tot for k, v in s.items()}

def random_matched(theta0, theta_ref, gen):
    delta = theta_ref - theta0; out = torch.zeros_like(delta)
    for (w, b) in [("W0", "b0"), ("W1", "b1"), ("W2", "b2")]:
        idx = torch.cat([torch.arange(SL[w].start, SL[w].stop), torch.arange(SL[b].start, SL[b].stop)])
        tau = delta[:, idx].norm(dim=1, keepdim=True)
        v = torch.randn(delta.shape[0], len(idx), generator=gen)
        out[:, idx] = v / (v.norm(dim=1, keepdim=True) + 1e-12) * tau
    return theta0 + out

def bias_matched(theta0, theta_ref, c):
    tau = (theta_ref - theta0).norm(dim=1); out = theta0.clone()
    out[torch.arange(len(c)), SL["b2"].start + torch.as_tensor(c)] -= tau
    return out

SANE_PT = {0: os.path.join(HERE, "sane_runs/s0/sane.pt"), 1: os.path.join(HERE, "sane_runs/s1/sane.pt")}
def get_psi(spec, seed, z, models):
    """editor (seed 0) / judge (seed 1) for a family; SANE psi is bound to the edit-set model ids."""
    if spec == "sane":
        from sane_psi import SanePsi
        return SanePsi(SANE_PT[seed], models) if os.path.exists(SANE_PT[seed]) else None
    p = NETS[(spec, seed)]
    return load_net(spec, p, z) if os.path.exists(p) else None

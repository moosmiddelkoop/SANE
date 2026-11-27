"""Flat MLP on the fc layer only (W_fc 100x512 + b_fc 100 = 51,300 inputs) of a staged ResNet-18 epoch zoo, paper recipe
(meta_cnn.MLPMeta / meta_cnn.train of the Small-CNN mlp_paper runs: 5 hidden layers x 256, ReLU, dropout 0.03, linear 100 + sigmoid,
MSE on the sigmoid output vs the 100-d per-class recall, AdamW lr 1e-3 wd 1e-5, OneCycle max_lr 1e-3 pct_start 0.05, batch 256,
30 epochs), with per-feature input standardisation (train-split mean/std + 1e-6) unless --no_std. Checkpoint selection: best val R2x
(double-centred; GMN Tier A criterion) evaluated every epoch. Test scoring = GMN Tier A metrics (train_gmn.py) + 2000 bootstrap.
usage: python train_mlpfc_ep20.py ZOO_ROOT OUT_DIR --seed S [--epochs 30] [--no_std] [--tag T] [--test_ids test_model_ids_ep20.npy]"""
import os, sys, json, time, math, argparse
sys.path.append(os.getcwd())
import numpy as np, torch, torch.nn as nn
import resnet_zoo as rz, edit_resnet as er
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--no_std", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--test_ids", default=None); ap.add_argument("--n_boot", type=int, default=2000); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--fc_dir", default=None, help="dir with zoo_meta.npz + fc_part_<start>.npy (rows of theta[:, fc]) instead of ROOT/zoo_theta.f32")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = rz.DEV; T0 = time.time()
name = f"mlpfc{a.tag}_s{a.seed}"; log = lambda *x: print(f"[{name} {time.time()-T0:6.0f}s]", *x, flush=True)

# ---- GMN Tier A metrics, copied verbatim from train_gmn.py
def dd(A): return A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
def r2x(Y, P):
    Yi, Pi = dd(Y), dd(P); sse = ((Yi - Pi) ** 2).sum(0); sst = (Yi ** 2).sum(0)
    return 1 - sse.sum() / sst.sum(), 1 - sse / sst
def r2raw(Y, P): return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
def r2model(Y, P):
    Yc = Y - Y.mean(1, keepdims=True); Pc = P - P.mean(1, keepdims=True)
    return 1 - ((Yc - Pc) ** 2).sum() / (Yc ** 2).sum()
def boot(Y, P, n, seed=0):
    rng = np.random.default_rng(seed); M = len(Y)
    s = np.array([r2x(Y[i], P[i])[0] for i in (rng.integers(0, M, M) for _ in range(n))])
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
def metrics(Y, P, n_boot=0):
    Y = np.asarray(Y, float); P = np.asarray(P, float); r, rk = r2x(Y, P)
    m = dict(r2x=float(r), r2_raw=float(r2raw(Y, P)), r2_model_centred=float(r2model(Y, P)),
             mse=float(((Y - P) ** 2).mean()), mae=float(np.abs(Y - P).mean()),
             r2x_per_class_median=float(np.median(rk)), r2x_per_class_frac_pos=float((rk > 0).mean()),
             pred_interaction_sd=float(dd(P).std()), true_interaction_sd=float(dd(Y).std()),
             corr_interaction=float(np.corrcoef(dd(Y).ravel(), dd(P).ravel())[0, 1]) if dd(P).std() > 0 else 0.0)
    if n_boot: m["r2x_ci95"] = boot(Y, P, n_boot)
    return m

meta = np.load(os.path.join(a.fc_dir or a.root, "zoo_meta.npz"), allow_pickle=True); M = int(meta["n_models"]); split = meta["split"]; R = meta["recall"].astype(np.float32)
if a.limit: M = a.limit; split = split[:M]; R = R[:M]
tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
if a.test_ids:
    ref = np.load(a.test_ids); assert np.array_equal(te, ref), "test split differs from the GMN study"
fc_idx = er.fc_param_indices().numpy(); lo, hi = int(fc_idx.min()), int(fc_idx.max()) + 1
assert hi - lo == len(fc_idx) == 51300 and (np.diff(fc_idx) == 1).all(), "fc indices not one contiguous block"
if a.fc_dir:
    import glob as _g
    parts = sorted(_g.glob(os.path.join(a.fc_dir, "fc_part_*.npy")), key=lambda f: int(f.rsplit("_", 1)[1][:-4]))
    X = np.concatenate([np.load(f) for f in parts])[:M].astype(np.float32); assert X.shape == (M, hi - lo), X.shape
else:
    theta = np.memmap(os.path.join(a.root, "zoo_theta.f32"), dtype=np.float32, mode="r", shape=(int(meta["n_models"]), rz.P))
    X = np.ascontiguousarray(theta[:M, lo:hi]).astype(np.float32); log("fc inputs", X.shape, f"{time.time()-T0:.0f}s")
mu = X[tr].mean(0) if not a.no_std else np.zeros(X.shape[1], np.float32)
sd = X[tr].std(0) + 1e-6 if not a.no_std else np.ones(X.shape[1], np.float32)

class FcMLP(nn.Module):           # = meta_cnn.MLPMeta(use_act=False) with D = 51,300, K = 100
    def __init__(self, D, K=100, hidden=256, n_hidden=5, dropout=0.03, mu=None, sd=None):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32)); self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        layers, d = [], D
        for _ in range(n_hidden): layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]; d = hidden
        layers += [nn.Linear(d, K)]; self.net = nn.Sequential(*layers)
    def logits(self, x): return self.net((x - self.mu) / self.sd)
    def forward(self, x): return torch.sigmoid(self.logits(x))

torch.manual_seed(a.seed); gen = torch.Generator().manual_seed(a.seed)
net = FcMLP(X.shape[1], mu=mu, sd=sd).to(dev); n_par = sum(p.numel() for p in net.parameters()); log("params", n_par)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5); batch = 256
spe = math.ceil(len(tr) / batch); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=a.epochs * spe, pct_start=0.05)
Xt = torch.from_numpy(X).to(dev); Yt = torch.from_numpy(R).to(dev)
@torch.no_grad()
def pred(ids): net.eval(); return net(Xt[ids]).cpu().numpy()
best, best_state, best_ep, hist = -1e9, None, 0, []
for ep in range(a.epochs):
    net.train(); perm = torch.randperm(len(tr), generator=gen); tot = 0.0
    for i in range(0, len(tr), batch):
        idx = torch.as_tensor(tr[perm[i:i + batch].numpy()], device=dev)
        L = ((torch.sigmoid(net.logits(Xt[idx])) - Yt[idx]) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step(); sched.step(); tot += L.item() * len(idx)
    v = float(r2x(R[va].astype(float), pred(va).astype(float))[0]); hist.append({"epoch": ep + 1, "train_mse": tot / len(tr), "val_r2x": v})
    if v > best: best, best_ep, best_state = v, ep + 1, {k: t.detach().clone() for k, t in net.state_dict().items()}
    if (ep + 1) % max(1, a.epochs // 10) == 0: log(f"ep {ep+1} train mse {tot/len(tr):.5f} val R2x {v:.4f}")
final_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
res = {"name": name, "args": vars(a), "params": int(n_par), "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), "hist": hist, "checkpoints": {}}
for ck, st, epn in (("best_val_r2x", best_state, best_ep), ("final", final_state, a.epochs)):
    net.load_state_dict(st); Pv, Pt = pred(va), pred(te)
    res["checkpoints"][ck] = {"epoch": epn, "val": metrics(R[va], Pv), "test": metrics(R[te], Pt, n_boot=a.n_boot)}
    np.save(os.path.join(a.out, f"{name}_{ck}_pred_test.npy"), Pt.astype(np.float32))
    log(ck, "epoch", epn, {k: res["checkpoints"][ck]["test"][k] for k in ("r2x", "r2x_ci95", "r2_raw", "mse", "mae")})
net.load_state_dict(best_state)
torch.save({"state_dict": net.state_dict(), "D": int(X.shape[1]), "fc_lo": lo, "fc_hi": hi, "best_epoch": best_ep, "args": vars(a)}, os.path.join(a.out, f"{name}.pt"))
np.save(os.path.join(a.out, "test_model_ids.npy"), te)
json.dump(res, open(os.path.join(a.out, f"{name}_results.json"), "w"), indent=1); log("DONE", f"{time.time()-T0:.0f}s")

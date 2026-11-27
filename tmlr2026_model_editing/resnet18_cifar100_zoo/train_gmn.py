"""Tier A: channel-graph metanet on the CIFAR-100 ResNet-18 zoo -- per-class recall prediction.
usage: python train_gmn.py ROOT OUTDIR --seed S [--epochs 300] [--reliability]
ROOT = stage_zoo.py layout (zoo_theta.f32, zoo_bn.npy, zoo_meta.npz, cifar100.npz).
Selection uses validation models only; test metrics are computed once, at the end, for three pre-declared checkpoints:
  'best_val_r2x' (headline), 'best_val_r2raw' (Small-CNN protocol), 'final'."""
import os, sys, json, time, math, argparse, copy
import numpy as np, torch
import resnet_zoo as rz
from resnet_gmn import ResNetGraphNet, norm_stats, normalise

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-5); ap.add_argument("--d", type=int, default=64)
ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--limit_train", type=int, default=0)
ap.add_argument("--reliability", action="store_true"); ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--device", default="cuda")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
T0 = time.time(); log = lambda *x: print(f"[{time.time()-T0:7.1f}s]", *x, flush=True)
dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
tag = f"gmn_d{a.d}_s{a.seed}_e{a.epochs}"

# ---------------------------------------------------------------- metrics
def dd(A): return A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
def r2x(Y, P):
    Yi, Pi = dd(Y), dd(P); sse = ((Yi - Pi) ** 2).sum(0); sst = (Yi ** 2).sum(0)
    return 1 - sse.sum() / sst.sum(), 1 - sse / sst
def r2raw(Y, P): return 1 - ((Y - P) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
def r2model(Y, P):  # model-centred: remove each model's own mean from truth and prediction
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

# ---------------------------------------------------------------- data
meta = np.load(os.path.join(a.root, "zoo_meta.npz"), allow_pickle=True)
M = int(meta["n_models"]); split = meta["split"]; R = meta["recall"].astype(np.float32)
tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
if a.limit_train: tr = tr[:a.limit_train]
need = np.concatenate([tr, va, te]); pos = {int(i): j for j, i in enumerate(need)}
log("models train/val/test", len(tr), len(va), len(te), "device", dev)
theta_mm = np.memmap(os.path.join(a.root, "zoo_theta.f32"), dtype=np.float32, mode="r", shape=(M, rz.P))
bn_all = np.load(os.path.join(a.root, "zoo_bn.npy"))

def to_dict(th, bf):
    p = rz.flat_to_params(th); p.update(rz.buffers_to_dict(bf))   # params (B,*shape), buffers (B,C)
    return p

# normalisation statistics from 100 training models
idx_st = np.sort(tr[:100])
st = norm_stats(to_dict(torch.from_numpy(np.ascontiguousarray(theta_mm[idx_st])), torch.from_numpy(bn_all[idx_st])))
json.dump({k: list(v) for k, v in st.items()}, open(os.path.join(a.out, f"{tag}_norm_stats.json"), "w"))   # needed to rebuild psi for editing
# load needed models, normalise per tensor, keep on device as one dict of (N, ...) tensors
keys = list(to_dict(torch.zeros(1, rz.P), torch.zeros(1, rz.NB)).keys())
store = None; CH = 50
for c0 in range(0, len(need), CH):
    ids = need[c0:c0 + CH]; srt = np.argsort(ids)
    th = torch.from_numpy(np.ascontiguousarray(theta_mm[ids[srt]]))[np.argsort(srt)]
    p = normalise(to_dict(th.to(dev), torch.from_numpy(bn_all[ids]).to(dev)), st)
    if store is None: store = {k: torch.empty((len(need),) + tuple(v.shape[1:]), dtype=torch.float32, device=dev) for k, v in p.items()}
    for k in keys: store[k][c0:c0 + len(ids)] = p[k]
    if c0 % 200 == 0: log(f"loaded {c0 + len(ids)}/{len(need)}")
log("data on device", sum(v.numel() for v in store.values()) * 4 / 1e9, "GB")
def batch(ids): j = torch.as_tensor([pos[int(i)] for i in ids], device=dev); return {k: v[j] for k, v in store.items()}
Rt = torch.from_numpy(R).to(dev)

# ---------------------------------------------------------------- optional: reliability of the interaction (split-half over test images)
if a.reliability:
    _, _, xte, yte = rz.load_cifar100(os.path.join(a.root, "cifar100.npz"))
    X = rz.prep_images(xte, "cifar100").to(dev); y = torch.from_numpy(yte).to(dev)
    g = torch.Generator().manual_seed(0); half = torch.zeros(len(y), dtype=torch.bool)
    for k in range(100):                                   # random 50/50 split of each class's 100 test images
        ii = torch.where(torch.from_numpy(yte) == k)[0]; half[ii[torch.randperm(len(ii), generator=g)[:len(ii) // 2]]] = True
    half = half.to(dev); RA, RB, RF = [], [], []
    for i in te:
        th = torch.from_numpy(np.array(theta_mm[i])).to(dev); bf = torch.from_numpy(bn_all[i]).to(dev)
        pr = rz.predict(th, bf, X); hit = pr == y
        rA = torch.bincount(y[hit & half], minlength=100) / torch.bincount(y[half], minlength=100)
        rB = torch.bincount(y[hit & ~half], minlength=100) / torch.bincount(y[~half], minlength=100)
        RA.append(rA.cpu().numpy()); RB.append(rB.cpu().numpy()); RF.append((torch.bincount(y[hit], minlength=100) / 100.).cpu().numpy())
    RA, RB, RF = map(np.array, (RA, RB, RF))
    rho_half = float(np.corrcoef(dd(RA).ravel(), dd(RB).ravel())[0, 1]); rho_full = 2 * rho_half / (1 + rho_half)
    # bootstrap over models for the split-half reliability
    rng = np.random.default_rng(1); bs = []
    for _ in range(1000):
        ii = rng.integers(0, len(te), len(te)); rh = np.corrcoef(dd(RA[ii]).ravel(), dd(RB[ii]).ravel())[0, 1]; bs.append(2 * rh / (1 + rh))
    # binomial-noise prediction of the interaction variance (independent Bernoulli per image)
    Yt = R[te].astype(float); noise_var = (Yt * (1 - Yt) / 100).mean() * (len(te) - 1) * 99 / (len(te) * 100)
    rel = dict(max_abs_recompute_vs_stored=float(np.abs(RF - R[te]).max()),
               interaction_var=float((dd(Yt) ** 2).mean()), binomial_noise_var=float(noise_var),
               split_half_r=rho_half, spearman_brown_reliability=rho_full,
               reliability_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
               note="split-half over a random 50/50 split of each class's 100 test images; reliability ~ ceiling on R2x for any predictor of test-set recall")
    json.dump(rel, open(os.path.join(a.out, "reliability_test_models.json"), "w"), indent=1); log("reliability", rel)
    del X

# ---------------------------------------------------------------- training
torch.manual_seed(a.seed); gen = torch.Generator().manual_seed(a.seed)
net = ResNetGraphNet(64, 100, a.d, fused=True, resid=True).to(dev)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
spe = math.ceil(len(tr) / a.batch)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * spe, pct_start=0.05)

@torch.no_grad()
def predict(ids, bs=64):
    net.eval(); out = []
    for i in range(0, len(ids), bs): out.append(net(batch(ids[i:i + bs])).float().cpu())
    net.train(); return torch.cat(out).numpy()

hist, best = [], {"best_val_r2x": (-1e9, None, 0), "best_val_r2raw": (-1e9, None, 0)}
for ep in range(1, a.epochs + 1):
    perm = tr[torch.randperm(len(tr), generator=gen).numpy()]; tot = 0.0
    for i in range(0, len(perm), a.batch):
        ids = perm[i:i + a.batch]
        L = ((net(batch(ids)) - Rt[torch.as_tensor(ids, device=dev)]) ** 2).mean()
        opt.zero_grad(set_to_none=True); L.backward()
        if a.clip: torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip)
        opt.step(); sched.step(); tot += L.item() * len(ids)
    Pv = predict(va); mv = metrics(R[va], Pv)
    h = dict(epoch=ep, train_mse=tot / len(tr), val_r2x=mv["r2x"], val_r2_raw=mv["r2_raw"], val_mse=mv["mse"],
             val_pred_int_sd=mv["pred_interaction_sd"], time=time.time() - T0); hist.append(h)
    for name, key in (("best_val_r2x", "val_r2x"), ("best_val_r2raw", "val_r2_raw")):
        if h[key] > best[name][0]: best[name] = (h[key], copy.deepcopy(net.state_dict()), ep)
    if ep == 1:   # exercise the full evaluation path on validation models once, so a late crash cannot lose the run
        json.dump(metrics(R[va], Pv, n_boot=50), open(os.path.join(a.out, f"{tag}_smoke_val.json"), "w"))
    if ep % 10 == 0 or ep <= 3:
        log(f"ep {ep} train_mse {h['train_mse']:.6f} val R2x {h['val_r2x']:+.4f} raw {h['val_r2_raw']:.4f} predIntSD {h['val_pred_int_sd']:.4f}")
        json.dump(hist, open(os.path.join(a.out, f"{tag}_history.json"), "w"))
best["final"] = (hist[-1]["val_r2x"], copy.deepcopy(net.state_dict()), a.epochs)
json.dump(hist, open(os.path.join(a.out, f"{tag}_history.json"), "w"))

# ---------------------------------------------------------------- test evaluation (once, three pre-declared checkpoints)
res = dict(tag=tag, args=vars(a), n_train=len(tr), n_val=len(va), n_test=len(te), graph_net_params=sum(t.numel() for t in net.parameters()),
           train_time_s=time.time() - T0, checkpoints={})
Yte = R[te]
for name, (vs, sd, ep) in best.items():
    net.load_state_dict(sd); Pt = predict(te); Pv = predict(va)
    np.save(os.path.join(a.out, f"{tag}_{name}_pred_test.npy"), Pt)
    res["checkpoints"][name] = dict(epoch=ep, val=metrics(R[va], Pv), test=metrics(Yte, Pt, n_boot=a.n_boot))
    if name == "best_val_r2x": torch.save(sd, os.path.join(a.out, f"{tag}_{name}.pt"))
    log(name, "epoch", ep, "test R2x", res["checkpoints"][name]["test"]["r2x"], res["checkpoints"][name]["test"]["r2x_ci95"])
# reference rows on the same test models: zoo-mean constant (train-split class means) and the true-interaction oracle check
res["reference"] = dict(zoo_mean_constant=metrics(Yte, np.tile(R[tr].mean(0), (len(te), 1))),
                        oracle_truth=metrics(Yte, Yte))
np.save(os.path.join(a.out, "test_model_ids.npy"), te)
json.dump(res, open(os.path.join(a.out, f"{tag}_results.json"), "w"), indent=1)
log("DONE", time.time() - T0)

"""Fit SANE property heads and baselines on the embeddings; write prediction table.
usage: python fit_heads.py EMB_NPZ FEATURES_NPZ OUT_PREFIX [--seed 0] [--epochs 300]
Heads: ridge on z̄ (linear probe), ridge on [z̄, per-layer means], MLP on z̄, token head on [z̄, z_fc(c)].
Baselines: ridge on weight statistics s(W), ridge on per-class fc-row statistics (+ class one-hot), ridge on both, zoo-mean constant.
Metrics on test models: R² (all cells), class-centred R² (per-class train means removed), MAE, median per-class R²."""
import os, sys, json, argparse
import numpy as np, torch, torch.nn as nn
ap = argparse.ArgumentParser(); ap.add_argument("emb"); ap.add_argument("feat"); ap.add_argument("out"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=300); ap.add_argument("--save_heads", default=None); a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
E = np.load(a.emb, allow_pickle=True); Fz = np.load(a.feat)
Y = E["recall"].astype(np.float64); split = E["split"]; NC = Y.shape[1]
tr, va, te = [np.where(split == s)[0] for s in ("train", "val", "test")]
zbar, zlay, zfc = E["zbar"].astype(np.float64), E["zlay"].astype(np.float64), E["zfc"].astype(np.float64)
ws, fcs = Fz["ws"].astype(np.float64), Fz["fcs"].astype(np.float64)
ws = np.nan_to_num(ws)
class_mean = Y[tr].mean(0)


def metrics(Yt, Pr):
    r2 = 1 - ((Yt - Pr) ** 2).sum() / ((Yt - Yt.mean()) ** 2).sum()
    Yc, Pc = Yt - class_mean, Pr - class_mean
    r2c = 1 - ((Yc - Pc) ** 2).sum() / ((Yc - Yc.mean()) ** 2).sum()
    pc = [1 - ((Yt[:, c] - Pr[:, c]) ** 2).sum() / max(((Yt[:, c] - Yt[:, c].mean()) ** 2).sum(), 1e-12) for c in range(NC)]
    return {"r2_all": float(r2), "r2_centred": float(r2c), "mae": float(np.abs(Yt - Pr).mean()), "r2_per_class_median": float(np.median(pc)),
            "r2_per_class_frac_pos": float(np.mean(np.array(pc) > 0)), "rmse": float(np.sqrt(((Yt - Pr) ** 2).mean()))}


def ridge(Xtr, Ytr, Xva, Yva, Xte, alphas=(1e-2, 1e-1, 1, 10, 100, 1e3, 1e4)):
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xs, Xv, Xt = (Xtr - m) / s, (Xva - m) / s, (Xte - m) / s
    ym = Ytr.mean(0); best = None
    for al in alphas:
        W = np.linalg.solve(Xs.T @ Xs + al * np.eye(Xs.shape[1]), Xs.T @ (Ytr - ym))
        pv = Xv @ W + ym; sc = metrics(Yva, pv)["r2_centred"] if Yva.ndim == 2 and Yva.shape[1] == NC else -((Yva - pv) ** 2).mean()
        if best is None or sc > best[0]: best = (sc, al, W)
    _, al, W = best
    return Xt @ W + ym, {"alpha": al, "val_score": best[0]}, (W, m, s, ym)


rows = []; preds = {}
def add(name, Pte, info=None, **extra):
    r = {"method": name, **metrics(Y[te], Pte), **(info or {}), **extra}; rows.append(r); preds[name] = Pte.astype(np.float32)
    print(f"{name:34s} r2_all {r['r2_all']:7.4f}  r2_centred {r['r2_centred']:7.4f}  mae {r['mae']:.4f}  medclass {r['r2_per_class_median']:7.4f}", flush=True)

add("zoo-mean constant", np.tile(class_mean, (len(te), 1)))
P_, info, _ = ridge(ws[tr], Y[tr], ws[va], Y[va], ws[te]); add("ridge weight statistics s(W)", P_, info, n_features=ws.shape[1])
P_, info, _ = ridge(zbar[tr], Y[tr], zbar[va], Y[va], zbar[te]); add("SANE linear probe on z̄", P_, info, n_features=zbar.shape[1])
ZL = np.concatenate([zbar, zlay.reshape(len(zbar), -1)], 1)
P_, info, _ = ridge(ZL[tr], Y[tr], ZL[va], Y[va], ZL[te]); add("SANE ridge on [z̄, layer means]", P_, info, n_features=ZL.shape[1])
# per-class stacked designs (shared ridge across classes with class one-hot)
def stacked(Fc, idx):  # Fc (M, NC, d) -> (len(idx)*NC, d+NC), y (len(idx)*NC)
    X = np.concatenate([Fc[idx].reshape(len(idx) * NC, -1), np.tile(np.eye(NC), (len(idx), 1))], 1)
    return X, Y[idx].reshape(-1)
Xtr_, ytr_ = stacked(fcs, tr); Xva_, yva_ = stacked(fcs, va); Xte_, _ = stacked(fcs, te)
P_, info, FCR = ridge(Xtr_, ytr_, Xva_, yva_, Xte_); add("ridge fc-row statistics (per class)", P_.reshape(len(te), NC), info, n_features=Xtr_.shape[1])
FW = np.concatenate([fcs, np.repeat(ws[:, None, :], NC, 1)], 2)
Xtr_, ytr_ = stacked(FW, tr); Xva_, yva_ = stacked(FW, va); Xte_, _ = stacked(FW, te)
P_, info, _ = ridge(Xtr_, ytr_, Xva_, yva_, Xte_); add("ridge fc-row stats + s(W)", P_.reshape(len(te), NC), info, n_features=Xtr_.shape[1])
ZF = np.concatenate([zfc.reshape(len(zfc), NC, -1), np.repeat(zbar[:, None, :], NC, 1)], 2)
Xtr_, ytr_ = stacked(ZF, tr); Xva_, yva_ = stacked(ZF, va); Xte_, _ = stacked(ZF, te)
P_, info, _ = ridge(Xtr_, ytr_, Xva_, yva_, Xte_); add("SANE ridge on [z̄, z_fc(c)] (per class)", P_.reshape(len(te), NC), info, n_features=Xtr_.shape[1])

# torch heads
dev = torch.device("cpu")
def fit_torch(make, Xtr, Xva, Xte, epochs, lr=1e-3, wd=1e-5, bs=32):
    net = make(); opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    Xtr_t, Xva_t, Xte_t = [torch.as_tensor(x, dtype=torch.float32) for x in (Xtr, Xva, Xte)]
    Ytr_t = torch.as_tensor(Y[tr], dtype=torch.float32)
    best, best_sd, hist = -1e9, None, []
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(tr))
        for s in range(0, len(tr), bs):
            i = perm[s:s + bs]; loss = ((net(Xtr_t[i]) - Ytr_t[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): sc = metrics(Y[va], net(Xva_t).numpy())["r2_centred"]
        hist.append(sc)
        if sc > best: best, best_sd = sc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_sd); net.eval()
    with torch.no_grad(): return net(Xte_t).numpy(), {"val_score": best, "best_epoch": int(np.argmax(hist)) + 1}, net

lat = zbar.shape[1]; zm, zs = zbar[tr].mean(0), zbar[tr].std(0) + 1e-8
Zn = (zbar - zm) / zs
class MLP(nn.Module):
    def __init__(s, d, h=512): super().__init__(); s.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, NC))
    def forward(s, x): return torch.sigmoid(s.net(x))
P_, info, mlp = fit_torch(lambda: MLP(lat), Zn[tr], Zn[va], Zn[te], a.epochs); add("SANE MLP head on z̄", P_, info)
class TokenHead(nn.Module):
    def __init__(s, d, h=512): super().__init__(); s.net = nn.Sequential(nn.Linear(d + NC, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(s, x):  # x (B, NC, d)
        B = x.shape[0]; return torch.sigmoid(s.net(torch.cat([x, torch.eye(NC)[None].expand(B, -1, -1)], 2))[..., 0])
fm, fs = ZF[tr].reshape(-1, ZF.shape[2]).mean(0), ZF[tr].reshape(-1, ZF.shape[2]).std(0) + 1e-8
ZFn = (ZF - fm) / fs
P_, info, tok = fit_torch(lambda: TokenHead(ZF.shape[2]), ZFn[tr], ZFn[va], ZFn[te], a.epochs); add("SANE token head [z̄, z_fc(c)]", P_, info)
# shuffled-model control: same token-head features but model identity permuted within the test set (expected r2_centred ~ 0)
perm = np.random.permutation(len(te)); add("control: token head, models shuffled", preds["SANE token head [z̄, z_fc(c)]"][perm])

import csv
with open(a.out + "_prediction.csv", "w", newline="") as f:
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "method", k)); w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
np.savez(a.out + "_pred_test.npz", test_idx=te, y_test=Y[te].astype(np.float32), class_mean=class_mean.astype(np.float32), **{k.replace(" ", "_"): v for k, v in preds.items()})
if a.save_heads:
    torch.save({"mlp": mlp.state_dict(), "token": tok.state_dict(), "zm": zm, "zs": zs, "fm": fm, "fs": fs, "class_mean": class_mean,
                "fcridge": {"W": FCR[0], "m": FCR[1], "s": FCR[2], "ym": FCR[3]}}, a.save_heads)
json.dump({"n_train": len(tr), "n_val": len(va), "n_test": len(te), "class_recall_sd_between_models_median": float(np.median(Y[tr].std(0))),
           "class_mean_range": [float(class_mean.min()), float(class_mean.max())]}, open(a.out + "_prediction_info.json", "w"), indent=1)
print("DONE")

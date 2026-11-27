"""Robustness of edits: relearning attack + linear-probe recoverability of the target class.
Reads results/<tag>.npz produced by run_edit.py with --save_delta, reconstructs theta1 = theta0 + delta for every
method in the file, and writes results/relearn_<tag>.npz with, per method and edit:
  probe_rec0_c / probe_rec1_c : recall of class c on the test set from a ridge-classifier probe fitted on
                                penultimate (GAP, 16-d) features of real train images, before / after the edit
  probe_rec1_c_dense          : same probe recall but using the edited model's OWN output layer on the frozen
                                penultimate features (= rec1_c, sanity check)
  relearn_k<k>_c, relearn_k<k>_ctrl : true target / mean-control recall after k Adam steps on a mixed batch
                                (n_forget class-c images + n_other images of the other classes), k in --ks
usage: python relearn_cnn.py results/<tag>.npz [--methods a,b] [--ks 1,5,20] [--n_forget 32] [--n_other 96] [--lr_mult 0.01]"""
import os, sys, json, time, argparse
os.environ["KMP_AFFINITY"] = "disabled"; os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.append(os.getcwd())
import numpy as np, torch, torch.nn.functional as F
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import cnnzoo, edit_cnn as ec

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("--methods", default="")
ap.add_argument("--ks", default="1,5,20"); ap.add_argument("--n_forget", type=int, default=32); ap.add_argument("--n_other", type=int, default=96)
ap.add_argument("--lr_mult", type=float, default=0.01); ap.add_argument("--n_probe_train", type=int, default=5000)
ap.add_argument("--probe_alpha", type=float, default=1.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--batch", type=int, default=100)
args = ap.parse_args()
tag = os.path.basename(args.path)[:-4]; outp = f"results/relearn_{tag}.npz"
if os.path.exists(outp) and not os.environ.get("FORCE_RERUN"):
    print(f"[{outp}] exists, skipping"); raise SystemExit(0)
t0 = time.time(); DEV = cnnzoo.DEV
ZOO = os.environ.get("ZOO", "mnist")
meta = np.load("zoo_meta.npz", allow_pickle=False); TH = np.load("zoo_theta.npy", mmap_mode="r"); M = len(meta["model"])
Xtr_u8, ytr, Xte_u8, yte = cnnzoo.load_images(ZOO)
Xte = cnnzoo.prep_images(Xte_u8).to(DEV); yte_t = torch.as_tensor(yte, device=DEV)
z = np.load(args.path, allow_pickle=False); jmeta = json.loads(str(z["meta"])); task = jmeta["task"]
methods = sorted({k.split("__")[0] for k in z.files if k.endswith("__delta")})
if args.methods:
    methods = [m for m in methods if m in args.methods.split(",")]
ks = sorted(int(k) for k in args.ks.split(","))
rng = np.random.default_rng(args.seed); gen = torch.Generator().manual_seed(args.seed)
probe_idx = rng.choice(len(ytr), args.n_probe_train, replace=False)
Xp = cnnzoo.prep_images(Xtr_u8[probe_idx]).to(DEV); yp = torch.as_tensor(ytr[probe_idx], device=DEV)
Yp = F.one_hot(yp, 10).double()


@torch.no_grad()
def gap_features(theta, act, X, chunk=None):
    chunk = chunk or max(32, 40000 // theta.shape[0])
    return torch.cat([cnnzoo.features(theta, X[i:i + chunk], act) for i in range(0, len(X), chunk)], 1)  # (B,N,16)


@torch.no_grad()
def probe_recall(theta, act, c):
    """Ridge-classifier probe on frozen GAP features: fit on real train images, recall of class c on test images."""
    Ftr = gap_features(theta, act, Xp).double()                          # (B,n,16)
    B = Ftr.shape[0]
    Ftr = torch.cat([Ftr, torch.ones(B, Ftr.shape[1], 1, dtype=Ftr.dtype, device=DEV)], -1)
    A = Ftr.transpose(1, 2) @ Ftr + args.probe_alpha * torch.eye(17, dtype=Ftr.dtype, device=DEV)
    W = torch.linalg.solve(A, Ftr.transpose(1, 2) @ Yp[None].expand(B, -1, -1))   # (B,17,10)
    Fte = gap_features(theta, act, Xte).double()
    Fte = torch.cat([Fte, torch.ones(B, Fte.shape[1], 1, dtype=Fte.dtype, device=DEV)], -1)
    pred = (Fte @ W).argmax(-1)                                           # (B,N)
    hit = (pred == yte_t[None]).double()
    rec = torch.zeros(B, 10, dtype=torch.double, device=DEV).index_add_(1, yte_t, hit) / torch.bincount(yte_t, minlength=10).double()
    return rec.gather(1, c[:, None])[:, 0].float()


def relearn(theta1, act, c_, ks):
    """Adam on CE for a mixed batch: n_forget images of class c_ + n_other of other classes (fixed per model). Returns dict k -> (rec_c, ctrl)."""
    B = theta1.shape[0]
    pool_c = np.where(ytr == c_)[0]; pool_o = np.where(ytr != c_)[0]
    idx = np.stack([np.concatenate([rng.choice(pool_c, args.n_forget, replace=False), rng.choice(pool_o, args.n_other, replace=False)]) for _ in range(B)])
    X = cnnzoo.prep_images(Xtr_u8[idx.reshape(-1)]).reshape(B, idx.shape[1], 1, *Xte.shape[-2:]).to(DEV)
    y = torch.as_tensor(ytr[idx], device=DEV)
    lr = args.lr_mult * theta1.pow(2).mean(1).sqrt()
    out = {}; th = theta1; done = 0
    for k in ks:
        th, _, _ = ec.finetune(th, act, X, y, lr, k - done, sign=+1, batch=idx.shape[1], gen=gen)   # full-batch steps
        done = k
        rec = ec.true_recall(th, act, Xte, yte_t)
        mask = torch.ones_like(rec, dtype=torch.bool); mask[:, c_] = False
        out[k] = (rec[:, c_].cpu().numpy(), ((rec * mask).sum(1) / 9).cpu().numpy())
    return out


res = {}
for m in methods:
    mi = z[f"{m}__model"].astype(int); cc = z[f"{m}__c"].astype(int); aa = z[f"{m}__act"].astype(int); delta = z[f"{m}__delta"]
    rec = {k: [] for k in ["probe_rec0_c", "probe_rec1_c", "probe_rec1_c_dense"] + [f"relearn_k{k}_{s}" for k in ks for s in ("c", "ctrl")] + ["relearn_k0_c", "relearn_k0_ctrl"]}
    for c_ in np.unique(cc):
        sel = np.where(cc == c_)[0]
        for s in range(0, len(sel), args.batch):
            ss = sel[s:s + args.batch]
            th0 = torch.as_tensor(np.ascontiguousarray(TH[2 * M + mi[ss]])).to(DEV); th1 = th0 + torch.as_tensor(delta[ss]).to(DEV)
            act = torch.as_tensor(aa[ss]).to(DEV); c = torch.full((len(ss),), int(c_), dtype=torch.long, device=DEV)
            rec["probe_rec0_c"].append(probe_recall(th0, act, c).cpu().numpy())
            rec["probe_rec1_c"].append(probe_recall(th1, act, c).cpu().numpy())
            r1 = ec.true_recall(th1, act, Xte, yte_t)
            mask = torch.ones_like(r1, dtype=torch.bool); mask[:, int(c_)] = False
            rec["probe_rec1_c_dense"].append(r1[:, int(c_)].cpu().numpy())
            rec["relearn_k0_c"].append(r1[:, int(c_)].cpu().numpy()); rec["relearn_k0_ctrl"].append(((r1 * mask).sum(1) / 9).cpu().numpy())
            rl = relearn(th1, act, int(c_), ks)
            for k in ks:
                rec[f"relearn_k{k}_c"].append(rl[k][0]); rec[f"relearn_k{k}_ctrl"].append(rl[k][1])
    for k, v in rec.items():
        res[f"{m}__{k}"] = np.concatenate(v)
    res[f"{m}__model"] = mi; res[f"{m}__c"] = cc
    print(f"[{tag}] {m}: n={len(mi)} probe rec c before {np.median(res[m+'__probe_rec0_c']):.3f} after {np.median(res[m+'__probe_rec1_c']):.3f} | "
          f"true after edit {np.median(res[m+'__relearn_k0_c']):.3f} -> " + " ".join(f"k{k} {np.median(res[m+f'__relearn_k{k}_c']):.3f}" for k in ks) + f"  ({time.time()-t0:.0f}s)", flush=True)
np.savez(outp, meta=json.dumps(dict(jmeta, ks=ks, n_forget=args.n_forget, n_other=args.n_other, lr_mult=args.lr_mult, methods=methods, time_s=time.time() - t0)), **res)
print(f"[{outp}] DONE {time.time()-t0:.0f}s")

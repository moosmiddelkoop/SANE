"""Relearning attack and linear-probe recoverability for edited ResNet-18 models.
usage: python relearn_resnet.py results/<tag>.npz [--n_per_class 1] [--ks 1,5,20] [--methods main,bias_matched,...]
For each selected edit and method, theta1 = theta0 + delta (fc / last-window / full delta as stored); reports true target & mean-control
recall after k Adam steps (lr 0.01*rms theta) on 32 class-c + 96 other train images (full model, BN eval), and the recall of class c
obtained by a ridge classifier on frozen GAP features (5,000 train images) before and after the edit. Writes results/relearn_<tag>.npz"""
import os, sys, json, time, argparse
sys.path.append(os.getcwd())
import numpy as np, torch
import resnet_zoo as rz, edit_resnet as er
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--n_per_class", type=int, default=1); ap.add_argument("--ks", default="1,5,20")
ap.add_argument("--methods", default=""); ap.add_argument("--n_probe", type=int, default=5000); ap.add_argument("--lr_mult", type=float, default=0.01); a = ap.parse_args()
ROOT = os.environ["ROOT"]; dev = rz.DEV; T0 = time.time(); tag = os.path.basename(a.npz)[:-4]
log = lambda *x: print(f"[relearn {tag} {time.time()-T0:5.0f}s]", *x, flush=True)
Rz = np.load(a.npz, allow_pickle=True); meta = json.loads(str(Rz["meta"])); task = meta["task"]
zoo = rz.Zoo(ROOT); norm, mp = meta["norm"], meta["maxpool"]
xtr, ytr, xte, yte = rz.load_cifar100(os.path.join(ROOT, "cifar100.npz")); Xte = rz.prep_images(xte, norm).to(dev); Yte = torch.from_numpy(yte).to(dev)
rp = np.random.default_rng(99); pidx_probe = np.concatenate([rp.choice(np.where(ytr == k)[0], a.n_probe // rz.NC, replace=False) for k in range(rz.NC)])
Xpr = rz.prep_images(xtr[pidx_probe], norm).to(dev); Ypr = torch.from_numpy(ytr[pidx_probe]).to(dev)
models, classes = Rz["model"], Rz["class"]; pidx_store = torch.from_numpy(Rz["pidx_store"]).to(dev)
sel = np.concatenate([np.where(classes == c)[0][: a.n_per_class] for c in np.unique(classes)])
ks = [int(k) for k in a.ks.split(",")]; gen = torch.Generator().manual_seed(5)
methods = [m for m in a.methods.split(",") if m] or (["main"] + [k[:-len("__rec1")] for k in Rz.files if k.endswith("__rec1") and not k.startswith("judge__") and not k.startswith("random_full")])
has_full = "delta_full16" in Rz.files and len(Rz["delta_full16"]) > 0
full_map = {int(m): i for i, m in enumerate(Rz["delta_full_model"])} if has_full else {}


def theta1_of(method, i, th0):
    """reconstruct the edited model for edit index i."""
    th = th0.clone(); c = int(classes[i])
    if method == "main":
        if meta["pidx_kind"] == "full":
            if int(models[i]) not in full_map: return None
            th += torch.from_numpy(Rz["delta_full16"][full_map[int(models[i])]].astype(np.float32)).to(dev)
        else:
            th[pidx_store] += torch.from_numpy(Rz["delta_fc16"][i].astype(np.float32)).to(dev)
    elif method == "bias_matched":
        th[rz.OFFSETS["fc.bias"][0] + c] += (-1 if task == "unlearn" else 1) * float(Rz["delta_norm"][i])
    elif method == "rowscale_matched":
        a0 = rz.OFFSETS["fc.weight"][0]; row = th[a0 + c * 512: a0 + (c + 1) * 512]; n = row.norm(); tau = float(Rz["delta_norm"][i])
        th[a0 + c * 512: a0 + (c + 1) * 512] = row * (max(0.0, 1 - tau / (n + 1e-12)) if task == "unlearn" else 1 + tau / (n + 1e-12))
    else:
        th[pidx_store] += torch.from_numpy(Rz[f"{method}__delta_fc16"][i].astype(np.float32)).to(dev)
    return th


@torch.no_grad()
def probe_recall(th, buf, c):
    """ridge classifier on GAP features (fit on 5,000 train images) -> recall of class c on the test set."""
    _, Ftr = rz.predict(th, buf, Xpr, maxpool=mp, features=True); _, Fte = rz.predict(th, buf, Xte, maxpool=mp, features=True)
    m, s = Ftr.mean(0), Ftr.std(0) + 1e-6; A = (Ftr - m) / s; Yoh = torch.nn.functional.one_hot(Ypr, rz.NC).float()
    W = torch.linalg.solve(A.T @ A + 10.0 * torch.eye(A.shape[1], device=dev), A.T @ (Yoh - Yoh.mean(0)))
    pred = (((Fte - m) / s) @ W + Yoh.mean(0)).argmax(1)
    return float((pred[Yte == c] == c).float().mean())


out = {m: {"rec_k": [], "probe0": [], "probe1": [], "rec1": [], "idx": []} for m in methods}
for n, i in enumerate(sel):
    mi, c = int(models[i]), int(classes[i]); th0 = zoo.rows([mi]).to(dev)[0]; buf = zoo.bufs([mi]).to(dev)[0]
    rr = np.random.default_rng(1000 + i); ii = np.concatenate([rr.choice(np.where(ytr == c)[0], 32, replace=False), rr.choice(np.where(ytr != c)[0], 96, replace=False)])
    X = rz.prep_images(xtr[ii], norm).to(dev)[None]; Y = torch.from_numpy(ytr[ii]).to(dev)[None]
    p0 = probe_recall(th0, buf, c) if n < len(sel) else None
    for m in methods:
        th1 = theta1_of(m, i, th0)
        if th1 is None: continue
        rec1 = rz.per_class_recall(th1[None], buf[None], Xte, Yte, maxpool=mp)[0][0]
        recs = []; th = th1[None]; done = 0
        for k in ks:
            th, _, _ = er.finetune(th, buf[None], X, Y, a.lr_mult * th0.pow(2).mean().sqrt(), k - done, sign=+1, batch=128, gen=gen, mp=mp); done = k
            recs.append(rz.per_class_recall(th, buf[None], Xte, Yte, maxpool=mp)[0][0].cpu())
        out[m]["rec_k"].append(torch.stack(recs)); out[m]["rec1"].append(rec1.cpu()); out[m]["probe0"].append(p0); out[m]["probe1"].append(probe_recall(th1, buf, c)); out[m]["idx"].append(i)
    if n % 10 == 0:
        mm = methods[0]; r = out[mm]["rec_k"][-1]
        log(f"{n}/{len(sel)} class {c}: {mm} target after edit {float(out[mm]['rec1'][-1][c]):.2f} -> k={ks}: {[round(float(x[c]), 2) for x in r]} probe {out[mm]['probe0'][-1]:.2f}->{out[mm]['probe1'][-1]:.2f}")
save = {"ks": np.array(ks), "class": classes, "model": models, "rec0": Rz["rec0"], "lr_mult": a.lr_mult}
for m, d in out.items():
    if d["idx"]:
        save[f"{m}__idx"] = np.array(d["idx"]); save[f"{m}__rec_k"] = torch.stack(d["rec_k"]).numpy(); save[f"{m}__rec1"] = torch.stack(d["rec1"]).numpy()
        save[f"{m}__probe0"] = np.array(d["probe0"], np.float32); save[f"{m}__probe1"] = np.array(d["probe1"], np.float32)
np.savez_compressed(f"results/relearn_{tag}.npz", **save); log("DONE", {m: len(d["idx"]) for m, d in out.items()}, f"{time.time()-T0:.0f}s")

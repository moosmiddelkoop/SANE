"""Re-run a parent metanet edit (Algorithm 1, unlearn, Boost beta=0.1, normalised step eta*||theta0||, psi frozen)
truncated per edit at its ORACLE step (first logged step where the stored TRUE recall_c on the 2000-image subset
is <= 0.4 x initial, along the walked path), and store delta_or = theta(oracle step) - theta0 for every reached edit.
Only edits still running are stepped (identical update rule to edit_cnn.edit: inactive rows are frozen there too).
Validation: true recall_c on the same 2000-image subset and psi_c at the reconstructed state vs. the stored
traj_true / traj_pred at the oracle step.
usage: python oracle_path.py <stored_results.npz> <key> <run> <out.npz> [--classes 0,1,..] [--max_models N]"""
import os, sys, json, time, argparse
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import cnnzoo, edit_cnn as ec

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("key"); ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9"); ap.add_argument("--eta", type=float, default=1e-3)
ap.add_argument("--restarts", type=int, default=8); ap.add_argument("--max_models", type=int, default=0)
args = ap.parse_args()
t0 = time.time(); DEV = cnnzoo.DEV
z = np.load(args.path, allow_pickle=False); K = args.key
d = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(K + "__")}
meta = np.load("zoo_meta.npz"); TH = np.load("zoo_theta.npy", mmap_mode="r"); M = len(meta["model"])
_, _, Xte_u8, yte = cnnzoo.load_images("mnist")
Xte = cnnzoo.prep_images(Xte_u8).to(DEV); yte_t = torch.as_tensor(yte.astype(np.int64), device=DEV)
_g = np.random.default_rng(12345)
_sub = np.concatenate([_g.choice(np.where(yte == k)[0], min(200, int((yte == k).sum())), replace=False) for k in range(10)])
Xs, ys = Xte[_sub], yte_t[_sub]

tt = d["traj_true"]; ts = d["traj_step"]; ts = ts if ts.ndim == 1 else np.nanmax(ts, 0)
st = d["steps"]; init = tt[:, 0]
crit = (tt <= 0.4 * init[:, None]) & (ts[None, :] <= st[:, None]) & ~np.isnan(tt)
reach = crit.any(1); fi = crit.argmax(1); s_or = np.where(reach, ts[fi], -1).astype(np.int64)

net = ec.load_metanet(f"runs/{args.run.split('@')[0]}.pt" + (("@" + args.run.split("@")[1]) if "@" in args.run else ""))
is_sane = hasattr(net, "set_perms")
if is_sane:
    net.restarts = args.restarts
    import sane_zoo as sz
mask = ec.output_layer_mask() if K == "meta_out" else None

N = len(st); delta_or = np.full((N, cnnzoo.D), np.nan, np.float32)
chk_true = np.full(N, np.nan, np.float32); chk_pred = np.full(N, np.nan, np.float32)
classes = [int(c) for c in args.classes.split(",")]
for c_ in classes:
    rows = np.where(d["c"] == c_)[0]
    if args.max_models:
        rows = rows[: args.max_models]
    mi = d["model"][rows]
    th0 = torch.as_tensor(np.ascontiguousarray(TH[2 * M + mi])).to(DEV)
    act = torch.as_tensor(d["act"][rows].astype(np.int64)).to(DEV)
    cc = torch.full((len(rows),), c_, dtype=torch.long, device=DEV)
    P0 = None
    if is_sane:
        with torch.no_grad():
            if getattr(net, "aligned", True):
                P0, _ = sz.weight_match_restarts(th0, net.theta_ref, restarts=args.restarts, seed=c_)
            else:
                P0 = sz.identity_perms(len(th0), th0.device)
    so = torch.as_tensor(s_or[rows], device=DEV)
    norm0 = th0.norm(dim=1, keepdim=True)
    theta = th0.clone()
    tc = time.time(); T = int(so.max().item()) if len(rows) else -1
    for t in range(max(T, 0)):
        a_idx = torch.where(so > t)[0]           # edits still running (oracle step not yet reached)
        if len(a_idx) == 0:
            break
        th = theta[a_idx].detach().requires_grad_(True)
        if is_sane:
            net.set_perms(P0[a_idx])
        pred = net(th, act[a_idx])
        f = ec.loss_fn(pred, cc[a_idx], "boost", 0.1, +1).sum()
        g, = torch.autograd.grad(f, th)
        if mask is not None:
            g = g * mask[None, :]
        with torch.no_grad():
            g = g / (g.norm(dim=1, keepdim=True) + 1e-12) * norm0[a_idx]
            theta[a_idx] = theta[a_idx] - args.eta * g
    r_ok = np.where(s_or[rows] >= 0)[0]
    if len(r_ok):
        with torch.no_grad():
            thr = theta[r_ok]
            rec = ec.true_recall(thr, act[r_ok], Xs, ys).gather(1, cc[r_ok][:, None])[:, 0].cpu().numpy()
            if is_sane:
                net.set_perms(P0[r_ok])
            pr = net(thr, act[r_ok]).gather(1, cc[r_ok][:, None])[:, 0].cpu().numpy()
        delta_or[rows[r_ok]] = (thr - th0[r_ok]).cpu().numpy()
        chk_true[rows[r_ok]] = rec; chk_pred[rows[r_ok]] = pr
    e_t = np.abs(chk_true[rows[r_ok]] - tt[rows[r_ok], fi[rows[r_ok]]]); e_p = np.abs(chk_pred[rows[r_ok]] - d["traj_pred"][rows[r_ok], fi[rows[r_ok]]])
    print(f"[{os.path.basename(args.out)}] class {c_}: n={len(rows)} reached={len(r_ok)} max_step={T} "
          f"|true-stored| med {np.median(e_t) if len(e_t) else np.nan:.4f} max {e_t.max() if len(e_t) else np.nan:.4f} "
          f"|psi-stored| med {np.median(e_p) if len(e_p) else np.nan:.4f} max {e_p.max() if len(e_p) else np.nan:.4f} ({time.time()-tc:.0f}s)", flush=True)
np.savez(args.out, model=d["model"], c=d["c"], act=d["act"], reach=reach, s_or=s_or, delta_or=delta_or,
         chk_true=chk_true, chk_pred=chk_pred, stored_true=np.where(reach, tt[np.arange(N), fi], np.nan),
         stored_pred=np.where(reach, d["traj_pred"][np.arange(N), fi], np.nan),
         meta=json.dumps(dict(path=args.path, key=K, run=args.run, eta=args.eta, classes=classes, time_s=time.time() - t0)))
print(f"[{args.out}] DONE {time.time()-t0:.0f}s", flush=True)

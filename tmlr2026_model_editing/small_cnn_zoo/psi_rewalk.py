"""Re-walk a stored metanet unlearning edit (Algorithm 1 as in edit_cnn.edit: Boost loss beta=0.1, normalised step
eta*||theta0||, psi frozen, relative psi-stop psi_c <= 0.4*psi_c(theta0) checked before every step, cap max_steps) with the
REAL psi-stop rule, for the same edits (model list/order of the stored results file), and save delta = theta_stop - theta0.
SANE editors: alignment perms from weight_match_restarts on the full 200-model class batch (restarts 8, seed = class),
held fixed along the path, exactly as run_edit.py. Only still-running edits are stepped (frozen rows are unchanged in
edit_cnn.edit too). Writes per edit: steps, stopped, pred0/pred1 (psi_c), rec1 (full-test per-class recall), delta.
usage: python psi_rewalk.py <stored.npz> <key> <run> <out.npz> --classes 0,1 [--max_steps 2000]"""
import os, sys, json, time, argparse
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import cnnzoo, edit_cnn as ec

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("key"); ap.add_argument("run"); ap.add_argument("out")
ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9"); ap.add_argument("--eta", type=float, default=1e-3)
ap.add_argument("--max_steps", type=int, default=2000); ap.add_argument("--restarts", type=int, default=8)
args = ap.parse_args()
t0 = time.time(); DEV = cnnzoo.DEV
z = np.load(args.path, allow_pickle=False); K = args.key
d = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(K + "__")}
meta = np.load("zoo_meta.npz"); TH = np.load("zoo_theta.npy", mmap_mode="r"); M = len(meta["model"])
_, _, Xte_u8, yte = cnnzoo.load_images("mnist")
Xte = cnnzoo.prep_images(Xte_u8).to(DEV); yte_t = torch.as_tensor(yte.astype(np.int64), device=DEV)
run = args.run
net = ec.load_metanet(f"runs/{run.split('@')[0]}.pt" + (("@" + run.split("@")[1]) if "@" in run else ""))
is_sane = hasattr(net, "set_perms")
if is_sane:
    net.restarts = args.restarts
    import sane_zoo as sz
mask = ec.output_layer_mask() if K == "meta_out" else None
classes = [int(c) for c in args.classes.split(",")]
res = {k: [] for k in ["idx", "model", "c", "steps", "stopped", "pred0", "pred1", "rec1", "delta"]}
for c_ in classes:
    rows = np.where(d["c"] == c_)[0]; mi = d["model"][rows]; B = len(rows)
    th0 = torch.as_tensor(np.ascontiguousarray(TH[2 * M + mi])).to(DEV)
    act = torch.as_tensor(d["act"][rows].astype(np.int64)).to(DEV); cc = torch.full((B,), c_, dtype=torch.long, device=DEV)
    P0 = None
    if is_sane:
        with torch.no_grad():
            P0 = sz.weight_match_restarts(th0, net.theta_ref, restarts=args.restarts, seed=c_)[0] if getattr(net, "aligned", True) \
                 else sz.identity_perms(B, th0.device)
        net.set_perms(P0)
    with torch.no_grad():
        p0 = net(th0, act).gather(1, cc[:, None])[:, 0]
    target = 0.4 * p0
    norm0 = th0.norm(dim=1, keepdim=True); theta = th0.clone()
    active = torch.ones(B, dtype=torch.bool, device=DEV); steps = torch.zeros(B, dtype=torch.long, device=DEV)
    tc = time.time()
    for t in range(args.max_steps + 1):
        a_idx = torch.where(active)[0]
        th = theta[a_idx].detach().requires_grad_(True)
        if is_sane:
            net.set_perms(P0[a_idx])
        pred = net(th, act[a_idx])
        pc = pred.gather(1, cc[a_idx][:, None])[:, 0].detach()
        done = pc <= target[a_idx]
        active[a_idx[done]] = False
        if t == args.max_steps or not active.any():
            break
        keep = ~done                                 # rows of a_idx that take a step
        f = ec.loss_fn(pred, cc[a_idx], "boost", 0.1, +1).sum()
        g, = torch.autograd.grad(f, th)
        if mask is not None:
            g = g * mask[None, :]
        with torch.no_grad():
            g = g / (g.norm(dim=1, keepdim=True) + 1e-12) * norm0[a_idx]
            upd = a_idx[keep]
            theta[upd] = theta[upd] - args.eta * g[keep]
        steps[upd] += 1
    with torch.no_grad():
        if is_sane:
            net.set_perms(P0)
        p1 = net(theta, act).gather(1, cc[:, None])[:, 0]
        r1 = ec.true_recall(theta, act, Xte, yte_t)
    for k, v in dict(idx=rows, model=mi, c=cc.cpu().numpy(), steps=steps.cpu().numpy(), stopped=(~active).cpu().numpy(),
                     pred0=p0.cpu().numpy(), pred1=p1.cpu().numpy(), rec1=r1.cpu().numpy(), delta=(theta - th0).cpu().numpy().astype(np.float32)).items():
        res[k].append(v)
    eq = (steps.cpu().numpy() == d["steps"][rows]).mean()
    print(f"[{os.path.basename(args.out)}] class {c_}: n={B} steps==stored {eq:.3f} |rec1_c-stored| med "
          f"{np.median(np.abs(r1.cpu().numpy()[:, c_] - d['rec1'][rows, c_])):.4f} ({time.time()-tc:.0f}s)", flush=True)
np.savez(args.out, **{k: np.concatenate(v, 0) for k, v in res.items()},
         meta=json.dumps(dict(path=args.path, key=K, run=run, eta=args.eta, max_steps=args.max_steps, classes=classes, time_s=time.time() - t0)))
print(f"[{args.out}] DONE {time.time()-t0:.0f}s", flush=True)

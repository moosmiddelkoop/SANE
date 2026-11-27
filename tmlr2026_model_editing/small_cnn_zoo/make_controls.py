"""Norm-matched controls for a parent metanet edit, at the parent's psi-stop or oracle-stop state.
Rules identical to run_edit.py derived baselines (edit_cnn.random_matched / bias_only_matched):
  random_matched: Gaussian vector rescaled to the parent's norm in each layer group (conv1 w+b, conv2 w+b, conv3 w+b, dense w+b),
                  torch.Generator seeded 0, drawn sequentially over classes 0..9 in the same group order;
  bias_matched:   b_c -= ||dtheta_parent|| (total norm).
Norm source per edit: 'delta' (saved parent delta), 'shares' (sqrt(share_group)*delta_norm from the stored per-layer
squared-norm shares; used only where no delta was saved) or 'oracle_delta' (reconstructed state from oracle_path.py).
Recall is evaluated on the full MNIST test set (as for every other row). steps/stopped are inherited from the parent state.
usage: python make_controls.py <parent_results.npz> <key> <state psi|oracle> <out.npz> [--oracle oracle.npz]"""
import os, sys, json, time, argparse
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import cnnzoo, edit_cnn as ec

ap = argparse.ArgumentParser()
ap.add_argument("path"); ap.add_argument("key"); ap.add_argument("state", choices=["psi", "oracle"]); ap.add_argument("out")
ap.add_argument("--oracle", default=""); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
t0 = time.time(); DEV = cnnzoo.DEV
z = np.load(args.path, allow_pickle=False); K = args.key
d = {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(K + "__")}
meta = np.load("zoo_meta.npz"); TH = np.load("zoo_theta.npy", mmap_mode="r"); M = len(meta["model"]); recall86 = meta["recall"][2]
_, _, Xte_u8, yte = cnnzoo.load_images("mnist")
Xte = cnnzoo.prep_images(Xte_u8).to(DEV); yte_t = torch.as_tensor(yte.astype(np.int64), device=DEV)
N = len(d["c"]); G = ec.LAYER_GROUPS; gidx = [ec.group_idx(w, b) for (w, b) in G]

# ---- per-edit group norms (N,4) and total norm (N,) at the parent's psi-stop state
if "delta" in d:
    D = torch.as_tensor(d["delta"].astype(np.float32))
    tau_psi = torch.stack([D[:, i].norm(dim=1) for i in gidx], 1).double().numpy(); tot_psi = D.norm(dim=1).double().numpy()
    route_psi = np.array(["delta"] * N); del D
else:
    sh = np.stack([d["share_conv1"], d["share_conv2"], d["share_conv3"],
                   d["share_dense_w_c"] + d["share_dense_w_other"] + d["share_dense_b_c"] + d["share_dense_b_other"]], 1).astype(np.float64)
    sh = np.clip(sh, 0, None)
    tau_psi = np.sqrt(sh) * d["delta_norm"][:, None].astype(np.float64); tot_psi = d["delta_norm"].astype(np.float64)
    route_psi = np.array(["shares"] * N)
steps = d["steps"].astype(np.int64); stopped = d["stopped"].astype(bool)
if args.state == "psi":
    tau, tot, route = tau_psi, tot_psi, route_psi
else:
    o = np.load(args.oracle)
    assert (o["model"] == d["model"]).all() and (o["c"] == d["c"]).all()
    reach = o["reach"].astype(bool); Do = torch.as_tensor(np.nan_to_num(o["delta_or"]))
    tau_o = torch.stack([Do[:, i].norm(dim=1) for i in gidx], 1).double().numpy(); tot_o = Do.norm(dim=1).double().numpy()
    tau = np.where(reach[:, None], tau_o, tau_psi); tot = np.where(reach, tot_o, tot_psi)
    route = np.where(reach, "oracle_delta", route_psi)
    steps = np.where(reach, o["s_or"], steps); stopped = np.where(reach, True, stopped)

gen = torch.Generator().manual_seed(args.seed)
rec = {n: {k: [] for k in ["model", "c", "act", "rec0", "rec1", "delta", "delta_norm", "theta_norm", "group_norms", "steps", "stopped",
                           "target_move", "Dt0", "Dt1", "Dr0", "Dr1", "ctrl_change", "worst_ctrl_move", "norm_route"]} for n in ("random_matched", "bias_matched")}
for c_ in range(10):
    rows = np.where(d["c"] == c_)[0]
    if len(rows) == 0:
        continue
    mi = d["model"][rows]
    th0 = torch.as_tensor(np.ascontiguousarray(TH[2 * M + mi]))
    act = torch.as_tensor(d["act"][rows].astype(np.int64)); cc = torch.full((len(rows),), c_, dtype=torch.long)
    tr = torch.as_tensor(tau[rows], dtype=torch.float32); tt_ = torch.as_tensor(tot[rows], dtype=torch.float32)
    dr = torch.zeros_like(th0)
    for gi, idx in enumerate(gidx):
        v = torch.randn(len(rows), len(idx), generator=gen)
        dr[:, idx] = v / (v.norm(dim=1, keepdim=True) + 1e-12) * tr[:, gi:gi + 1]
    db = torch.zeros_like(th0); db[torch.arange(len(rows)), ec.SEG["db"][0] + cc] = -tt_
    rec0 = torch.as_tensor(recall86[mi])
    for name, dl in (("random_matched", dr), ("bias_matched", db)):
        th1 = (th0 + dl).to(DEV)
        with torch.no_grad():
            r1 = ec.true_recall(th1, act.to(DEV), Xte, yte_t).cpu()
        sc = ec.scores(rec0, r1, cc, "unlearn")
        R = rec[name]
        for k, v in dict(model=mi, c=cc.numpy(), act=act.numpy(), rec0=rec0.numpy(), rec1=r1.numpy(), delta=dl.numpy().astype(np.float32),
                         delta_norm=dl.norm(dim=1).numpy(), theta_norm=th0.norm(dim=1).numpy(), group_norms=tau[rows], steps=steps[rows],
                         stopped=stopped[rows], norm_route=route[rows], **{k2: v2.numpy() for k2, v2 in sc.items()}).items():
            R[k].append(v)
    print(f"[{os.path.basename(args.out)}] class {c_} n={len(rows)} ({time.time()-t0:.0f}s)", flush=True)
out = {f"{n}__{k}": np.concatenate(v, 0) for n, R in rec.items() for k, v in R.items()}
jm = json.dumps(dict(method=f"controls:{os.path.basename(args.path)}:{K}", parent=args.path, parent_key=K, state=args.state, oracle=args.oracle,
                     task="unlearn", set="test", seed=args.seed, n=N, time_s=time.time() - t0))
np.savez_compressed(args.out, meta=jm, **out)
print(f"[{args.out}] DONE {time.time()-t0:.0f}s", flush=True)

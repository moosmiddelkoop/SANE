"""Norm-matched controls for the free-walk oracle states (same construction as make_controls.py / run_edit.py derived
baselines): random_matched = Gaussian vector rescaled to the parent's norm in each layer group (conv1/conv2/conv3/dense,
w+b), torch generator seed 0 drawn class by class 0..9 in group order; bias_matched = b_c -= ||dtheta_parent||.
Norms from the saved free-walk delta of every edit. Recall on the full MNIST test set; steps inherited (oracle step or cap).
usage: python make_controls_fw.py <freewalk_merged.npz> <out.npz>"""
import os, sys, json, time
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import cnnzoo, edit_cnn as ec
fin, fout = sys.argv[1], sys.argv[2]
t0 = time.time(); DEV = cnnzoo.DEV
P = np.load(fin)
meta = np.load("zoo_meta.npz"); TH = np.load("zoo_theta.npy", mmap_mode="r"); M = len(meta["model"]); recall86 = meta["recall"][2]
_, _, Xte_u8, yte = cnnzoo.load_images("mnist")
Xte = cnnzoo.prep_images(Xte_u8).to(DEV); yte_t = torch.as_tensor(yte.astype(np.int64), device=DEV)
gidx = [ec.group_idx(w, b) for (w, b) in ec.LAYER_GROUPS]
D = torch.as_tensor(P["delta"].astype(np.float32))
tau = torch.stack([D[:, i].norm(dim=1) for i in gidx], 1); tot = D.norm(dim=1); del D
gen = torch.Generator().manual_seed(0)
rec = {n: {k: [] for k in ["model", "c", "act", "rec0", "rec1", "delta", "delta_norm", "steps", "reached", "target_move", "Dt0", "Dt1", "Dr0", "Dr1", "ctrl_change", "worst_ctrl_move"]}
       for n in ("random_matched", "bias_matched")}
for c_ in range(10):
    rows = np.where(P["c"] == c_)[0]; mi = P["model"][rows]
    th0 = torch.as_tensor(np.ascontiguousarray(TH[2 * M + mi])); act = torch.as_tensor(P["act"][rows].astype(np.int64)); cc = torch.full((len(rows),), c_, dtype=torch.long)
    dr = torch.zeros_like(th0)
    for gi, idx in enumerate(gidx):
        v = torch.randn(len(rows), len(idx), generator=gen)
        dr[:, idx] = v / (v.norm(dim=1, keepdim=True) + 1e-12) * tau[rows, gi:gi + 1]
    db = torch.zeros_like(th0); db[torch.arange(len(rows)), ec.SEG["db"][0] + cc] = -tot[rows]
    rec0 = torch.as_tensor(recall86[mi])
    for name, dl in (("random_matched", dr), ("bias_matched", db)):
        with torch.no_grad():
            r1 = ec.true_recall((th0 + dl).to(DEV), act.to(DEV), Xte, yte_t).cpu()
        sc = ec.scores(rec0, r1, cc, "unlearn")
        for k, v in dict(model=mi, c=cc.numpy(), act=act.numpy(), rec0=rec0.numpy(), rec1=r1.numpy(), delta=dl.numpy().astype(np.float32),
                         delta_norm=dl.norm(dim=1).numpy(), steps=P["steps"][rows], reached=P["reached"][rows], **{k2: v2.numpy() for k2, v2 in sc.items()}).items():
            rec[name][k].append(v)
    print(f"[{os.path.basename(fout)}] class {c_} ({time.time()-t0:.0f}s)", flush=True)
np.savez_compressed(fout, meta=json.dumps(dict(parent=fin, seed=0, task="unlearn", set="test", time_s=time.time() - t0)),
                    **{f"{n}__{k}": np.concatenate(v, 0) for n, R in rec.items() for k, v in R.items()})
print(f"[{fout}] DONE {time.time()-t0:.0f}s", flush=True)

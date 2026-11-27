"""Per-layer norm-matched random-vector control (edit_resnet.random_layer_matched) for the STORED graph-net full-gradient edits of a
run_edit_gmn.py result file, using the exact stored parent deltas (delta_full16: the first edit of each class in every 40-edit batch).
No re-walk: the editor checkpoint no longer exists. For every stored row r -> edit index e (reconstructed exactly as in relearn_resnet_fixidx.py):
  tau_g = ||delta_g|| (21 groups), v ~ N(0, I) from torch.Generator(seed = SEED_BASE + e) on the CPU, RV = theta0 + concat_g v_g/||v_g|| tau_g;
  full-test per-class recall of theta0 + RV, of theta0 + delta (parent reconstruction check), group/total-norm checks, ||RV||^2 block shares.
--relearn: relearning attack on the RV for the first edit of each class (= relearn_resnet_fixidx.py edit-index-keyed selection, n_per_class 1): same images
(rng 1000+i), Adam lr lr_mult*rms(theta0), k = 1,5,20 cumulative, full-test recall; finetune generator torch.Generator(5) as in relearn_resnet_fixidx.py.
usage: ROOT=<zoo> python build_layer_rv_ep20.py EDIT_NPZ OUT_NPZ [--relearn] [--seed_base 20261112] [--limit N]"""
import os, sys, json, time, argparse
sys.path.append(os.getcwd()); sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import resnet_zoo as rz, edit_resnet as er
ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("out"); ap.add_argument("--relearn", action="store_true")
ap.add_argument("--seed_base", type=int, default=20261112); ap.add_argument("--ks", default="1,5,20"); ap.add_argument("--lr_mult", type=float, default=1e-3)
ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
ROOT = os.environ["ROOT"]; dev = rz.DEV; T0 = time.time(); log = lambda *x: print(f"[layer_rv {time.time()-T0:6.0f}s]", *x, flush=True)
Rz = np.load(a.npz, allow_pickle=True); meta = json.loads(str(Rz["meta"])); args = meta["args"]; norm, mp = meta["norm"], meta["maxpool"]
assert meta["pidx_kind"] == "full"
models, classes = Rz["model"], Rz["class"].astype(int); dfm = Rz["delta_full_model"]
bs, dn = int(args["batch"]), int(args["delta_full_n"]); rows = []
for s in range(0, len(models), bs):
    cb = classes[s:s + bs]; rows += list(s + np.concatenate([np.where(cb == ci)[0][:dn] for ci in np.unique(cb)]))
rows = np.array(rows); assert len(rows) == len(dfm) and (models[rows] == dfm).all(), "row -> edit map mismatch"
log("stored full deltas", len(rows), "of", len(models), "edits; loading delta_full16 ...")
D16 = Rz["delta_full16"]; log("loaded", D16.shape, D16.dtype)
zoo = rz.Zoo(ROOT); xtr, ytr, xte, yte = rz.load_cifar100(os.path.join(ROOT, "cifar100.npz"))
Xte = rz.prep_images(xte, norm).to(dev); Yte = torch.from_numpy(yte).to(dev)
n = len(rows) if not a.limit else min(a.limit, len(rows))
out = {k: [] for k in ["edit_idx", "model", "class", "seed", "rec1_rv", "rec1_parent_recon", "gnorm_parent", "gnorm_rv", "tot_parent", "tot_rv",
                       "share_fc_w_c", "share_fc_w_other", "share_fc_b_c", "share_fc_b_other", "share_conv"]}
def rv_of(e, th0, d):
    return er.random_layer_matched(th0[None], d[None], gen=torch.Generator().manual_seed(a.seed_base + int(e)))[0]
with torch.no_grad():
    for r in range(n):
        e = int(rows[r]); m, c = int(models[e]), int(classes[e]); th0 = zoo.rows([m]).to(dev)[0]; buf = zoo.bufs([m]).to(dev)[0]
        d = torch.from_numpy(D16[r].astype(np.float32)).to(dev); th_rv = rv_of(e, th0, d); drv = th_rv - th0
        out["rec1_rv"].append(rz.per_class_recall(th_rv[None], buf[None], Xte, Yte, maxpool=mp)[0][0].cpu().numpy())
        out["rec1_parent_recon"].append(rz.per_class_recall((th0 + d)[None], buf[None], Xte, Yte, maxpool=mp)[0][0].cpu().numpy())
        out["gnorm_parent"].append(er.layer_group_norms(d[None])[0].cpu().numpy()); out["gnorm_rv"].append(er.layer_group_norms(drv[None])[0].cpu().numpy())
        out["tot_parent"].append(float(d.norm())); out["tot_rv"].append(float(drv.norm()))
        sh = er.delta_shares(drv[None], torch.tensor([c], device=dev))
        for k in ["share_fc_w_c", "share_fc_w_other", "share_fc_b_c", "share_fc_b_other", "share_conv"]: out[k].append(float(sh[k][0]))
        out["edit_idx"].append(e); out["model"].append(m); out["class"].append(c); out["seed"].append(a.seed_base + e)
        if r % 20 == 0: log(f"{r}/{n} edit {e} class {c}: target rec0 {Rz['rec0'][e, c]:.2f} parent {out['rec1_parent_recon'][-1][c]:.2f} (stored {Rz['rec1'][e, c]:.2f}) RV {out['rec1_rv'][-1][c]:.2f}")
save = {k: np.array(v) for k, v in out.items()}
ge = np.abs(save["gnorm_rv"] - save["gnorm_parent"]) / np.maximum(save["gnorm_parent"], 1e-12); te = np.abs(save["tot_rv"] - save["tot_parent"]) / save["tot_parent"]
log("max rel err group norm", float(ge.max()), "total", float(te.max()))
save.update(group_names=np.array(er.LAYER_GROUP_NAMES), rec0=Rz["rec0"][save["edit_idx"]], rec1_stored=Rz["rec1"][save["edit_idx"]], delta_norm_stored=Rz["delta_norm"][save["edit_idx"]],
            stopped=Rz["stopped"][save["edit_idx"]], steps=Rz["steps"][save["edit_idx"]],
            oracle_step=(Rz["oracle_step"][save["edit_idx"]] if "oracle_step" in Rz.files else np.full(n, -9)),
            max_rel_err_group=float(ge.max()), max_rel_err_total=float(te.max()), seed_base=a.seed_base,
            meta=json.dumps({"src": a.npz, "parent_args": args, "n_rows_stored": int(len(rows)), "seed_rule": "torch.Generator().manual_seed(seed_base + edit_index), CPU randn over all P"}))
if a.relearn:
    ks = [int(k) for k in a.ks.split(",")]; sel = np.concatenate([np.where(classes == cc)[0][:1] for cc in np.unique(classes)])
    pos = {int(e): r for r, e in enumerate(rows)}; assert all(int(i) in pos for i in sel), "a relearn edit has no stored delta"
    gen = torch.Generator().manual_seed(5); rk_all, r1_all, idx_all = [], [], []
    for nn_, i in enumerate(sel):
        mi, c = int(models[i]), int(classes[i]); th0 = zoo.rows([mi]).to(dev)[0]; buf = zoo.bufs([mi]).to(dev)[0]
        rr = np.random.default_rng(1000 + int(i)); ii = np.concatenate([rr.choice(np.where(ytr == c)[0], 32, replace=False), rr.choice(np.where(ytr != c)[0], 96, replace=False)])
        X = rz.prep_images(xtr[ii], norm).to(dev)[None]; Y = torch.from_numpy(ytr[ii]).to(dev)[None]
        d = torch.from_numpy(D16[pos[int(i)]].astype(np.float32)).to(dev)
        with torch.no_grad(): th1 = rv_of(i, th0, d); rec1 = rz.per_class_recall(th1[None], buf[None], Xte, Yte, maxpool=mp)[0][0]
        recs = []; th = th1[None]; done = 0
        for k in ks:
            th, _, _ = er.finetune(th, buf[None], X, Y, a.lr_mult * th0.pow(2).mean().sqrt(), k - done, sign=+1, batch=128, gen=gen, mp=mp); done = k
            recs.append(rz.per_class_recall(th, buf[None], Xte, Yte, maxpool=mp)[0][0].cpu())
        rk_all.append(torch.stack(recs).numpy()); r1_all.append(rec1.cpu().numpy()); idx_all.append(int(i))
        if nn_ % 20 == 0: log(f"relearn {nn_}/{len(sel)} class {c}: RV target {float(rec1[c]):.2f} -> {[round(float(x[c]), 2) for x in recs]}")
    save.update(rl_idx=np.array(idx_all), rl_rec1=np.array(r1_all), rl_rec_k=np.array(rk_all), rl_ks=np.array(ks), rl_lr_mult=a.lr_mult)
np.savez_compressed(a.out, **save); log("DONE", n, "edits", f"{time.time()-T0:.0f}s")

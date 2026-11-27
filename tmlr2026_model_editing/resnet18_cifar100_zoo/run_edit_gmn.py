"""Editing driver for the ResNet-18 zoo with the channel-graph metanet as psi (same result format as run_edit_resnet.py).
usage: python run_edit_gmn.py SPEC TASK [--set test|val] [--n_per_class 6] [--eta 1e-3] [--max_steps 1000] [--log_every 50] [--batch 40]
         [--judges gmn@s1] [--tag T] [--no_derived] [--delta_full_n 1] [--beta 0.1] [--lr_mult 1e-3]
SPEC: meta:gmn (full gradient) | meta_fc:gmn (gradient masked to the fc layer) | ga_real | ft_retain
env: ROOT = staged zoo dir (zoo_theta.f32, zoo_bn.npy, zoo_meta.npz, cifar100.npz); RUNS = graph-net runs root (RUNS/s<seed>/...);
     EDITOR_SEED = seed of the editing graph net. Files in cwd: edit_sets.npz. Writes results/<tag>.npz"""
import os, sys, json, time, argparse
sys.path.append(os.getcwd()); sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import resnet_zoo as rz, edit_resnet as er
from gmn_psi import load_gmn_psi
ap = argparse.ArgumentParser(); ap.add_argument("spec"); ap.add_argument("task"); ap.add_argument("--set", default="test")
ap.add_argument("--n_per_class", type=int, default=6); ap.add_argument("--eta", type=float, default=1e-3)
ap.add_argument("--max_steps", type=int, default=1000); ap.add_argument("--log_every", type=int, default=50); ap.add_argument("--batch", type=int, default=40)
ap.add_argument("--judges", default=""); ap.add_argument("--tag", default=None); ap.add_argument("--no_derived", action="store_true")
ap.add_argument("--delta_full_n", type=int, default=1); ap.add_argument("--classes", default=""); ap.add_argument("--beta", type=float, default=0.1)
ap.add_argument("--lr_mult", type=float, default=1e-3); ap.add_argument("--time_limit", type=float, default=None)
ap.add_argument("--ga_steps", type=int, default=200); ap.add_argument("--ft_steps", type=int, default=100)
ap.add_argument("--fixed_steps", action="store_true", help="ignore psi's stop rule: walk max_steps for every edit (oracle-stop diagnostic)")
ap.add_argument("--oracle_stop", action="store_true", help="ignore psi's stop; freeze each edit when TRUE recall_c (2k-image subset, every log_every steps) meets the criterion")
a = ap.parse_args()
ROOT = os.environ["ROOT"]; RUNS = os.environ.get("RUNS", ""); SEED = int(os.environ.get("EDITOR_SEED", 0)); dev = rz.DEV; T0 = time.time()
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
tag = a.tag or f"{a.spec.replace(':', '_')}_{a.task}_{a.set}"
os.makedirs("results", exist_ok=True)
if os.path.exists(f"results/{tag}.npz") and not os.environ.get("FORCE_RERUN"): print(f"[{tag}] exists, skipping"); raise SystemExit(0)
log = lambda *x: print(f"[{tag} {time.time()-T0:6.0f}s]", *x, flush=True)
zoo = rz.Zoo(ROOT); norm, mp = "cifar100", False      # preprocessing verified against stored acc_class in the Tier A reliability check
xtr, ytr, xte, yte = rz.load_cifar100(os.path.join(ROOT, "cifar100.npz"))
Xte = rz.prep_images(xte, norm).to(dev); Yte = torch.from_numpy(yte).to(dev)
g = np.random.default_rng(12345); sub = np.concatenate([g.choice(np.where(yte == k)[0], min(20, (yte == k).sum()), replace=False) for k in range(rz.NC)])
Xsub, Ysub = Xte[sub], Yte[sub]
gen = torch.Generator().manual_seed(1000 + SEED)
LAYER_RV_SEED = 20261112 + 100000 * SEED   # per-batch seed = LAYER_RV_SEED + index of the first edit of the batch

# edit set
ES = np.load("edit_sets.npz"); key = f"{a.task}_{a.set}"
models, classes = ES[key + "_model"], ES[key + "_class"]
if a.classes:
    keep = np.isin(classes, [int(x) for x in a.classes.split(",")]); models, classes = models[keep], classes[keep]
order = np.lexsort((models, classes)); models, classes = models[order], classes[order]
sel = np.concatenate([np.where(classes == c)[0][: a.n_per_class] for c in np.unique(classes)]); models, classes = models[sel], classes[sel]
log(f"{len(models)} edits over {len(np.unique(classes))} classes; editor seed {SEED}; preprocessing {norm} maxpool={mp}")


def load_psi(spec_name, seed=SEED):
    if spec_name == "gmn": return load_gmn_psi(RUNS, seed, dev=dev)
    raise ValueError(spec_name)


kind, _, name = a.spec.partition(":")
psi = load_psi(name) if name else None
judges = {}
for j in [x for x in a.judges.split(",") if x]:
    jn, _, js = j.partition("@"); judges[j] = load_psi(jn, int(js[1:]) if js else SEED)
fc_idx = er.fc_param_indices().to(dev)
if kind == "meta_fc": pidx = fc_idx
elif kind in ("meta", "ga_real", "ft_retain"): pidx = None
else: raise ValueError(kind)
pidx_store = fc_idx if pidx is None else pidx

rec_fn = lambda th, bufs: rz.per_class_recall(th, bufs, Xte, Yte, maxpool=mp)[0]
R = {k: [] for k in ["model", "class", "steps", "stopped", "psi_stop_step", "oracle_step", "pred0", "pred1", "rec0", "rec1", "delta_norm", "delta_fc16", "traj_pred", "traj_true", "traj_mon",
                     "share_fc_w_c", "share_fc_w_other", "share_fc_b_c", "share_fc_b_other", "share_conv", "sign_b_c", "delta_full16", "delta_full_model"]}
D = {}   # derived / control methods: name -> dict of lists
def add_derived(nm, th1, th0, bufs, pred_fn=None):
    d = D.setdefault(nm, {"rec1": [], "pred1": [], "delta_norm": [], "delta_fc16": []})
    d["rec1"].append(rec_fn(th1, bufs).cpu()); d["delta_norm"].append((th1 - th0).norm(dim=1).cpu()); d["delta_fc16"].append((th1 - th0)[:, pidx_store].half().cpu())
    with torch.no_grad():
        d["pred1"].append((pred_fn(th1) if pred_fn is not None else torch.full((th1.shape[0], rz.NC), float("nan"), device=dev)).cpu())
J = {j: {"pred0": [], "pred1": []} for j in judges}
traj_step = None
xtr_t = torch.from_numpy(xtr); ytr_t = torch.from_numpy(ytr)

for s in range(0, len(models), a.batch):
    mb, cb = models[s:s + a.batch], classes[s:s + a.batch]; B = len(mb)
    th0 = zoo.rows(mb).to(dev); bufs = zoo.bufs(mb).to(dev); c = torch.from_numpy(cb).to(dev); ar = torch.arange(B, device=dev)
    rec0 = rec_fn(th0, bufs)
    eval_fn = lambda th: rz.per_class_recall(th, bufs, Xsub, Ysub, maxpool=mp)[0][ar, c]
    if psi is not None: psi.set_batch(bufs)
    for j in judges.values(): j.set_batch(bufs)
    t_b = time.time()
    if kind in ("meta", "meta_fc"):
        out = er.edit(th0, c, psi, task=a.task, beta=a.beta, eta=a.eta, max_steps=a.max_steps, pidx=pidx, eval_fn=eval_fn, log_every=a.log_every, use_stop=not (a.fixed_steps or a.oracle_stop), oracle_stop=a.oracle_stop)
    elif kind in ("ga_real", "ft_retain"):
        rms = th0.pow(2).mean(1).sqrt(); lr = a.lr_mult * rms
        if kind == "ga_real":
            X, Y, Xh, Yh = [], [], [], []
            for ci in cb:
                ii = np.random.default_rng(int(ci) + 7).permutation(np.where(ytr == ci)[0]); X.append(rz.prep_images(xtr[ii[:256]], norm)); Y.append(ytr_t[ii[:256]])
                Xh.append(rz.prep_images(xtr[ii[256:456]], norm)); Yh.append(ytr_t[ii[256:456]])
            X, Y, Xh, Yh = torch.stack(X).to(dev), torch.stack(Y).to(dev), torch.stack(Xh).to(dev), torch.stack(Yh).to(dev)
            with torch.no_grad():
                r0h = torch.stack([(rz.predict(th0[i], bufs[i], Xh[i], maxpool=mp) == Yh[i]).float().mean() for i in range(B)])
            def stop_fn(th):
                r = torch.stack([(rz.predict(th[i], bufs[i], Xh[i], maxpool=mp) == Yh[i]).float().mean() for i in range(B)])
                return r <= 0.4 * r0h if a.task == "unlearn" else r >= 0.9
            th1, used, stopped = er.finetune(th0, bufs, X, Y, lr, a.ga_steps, sign=(-1 if a.task == "unlearn" else +1), gen=gen, stop_fn=stop_fn, mp=mp)
        else:
            X, Y = [], []
            for ci in cb:
                rr = np.random.default_rng(int(ci) + 11)
                if a.task == "unlearn":
                    ii = rr.choice(np.where(ytr != ci)[0], 256, replace=False)
                else:
                    ii = np.concatenate([rr.choice(np.where(ytr == ci)[0], 128, replace=False), rr.choice(np.where(ytr != ci)[0], 128, replace=False)])
                X.append(rz.prep_images(xtr[ii], norm)); Y.append(ytr_t[ii])
            X, Y = torch.stack(X).to(dev), torch.stack(Y).to(dev)
            th1, used, stopped = er.finetune(th0, bufs, X, Y, lr, a.ft_steps, sign=+1, gen=gen, mp=mp)
        out = dict(theta=th1, steps=used, stopped=stopped, pred0=torch.full((B, rz.NC), float("nan"), device=dev), pred1=torch.full((B, rz.NC), float("nan"), device=dev))
    th1 = out["theta"]; delta = th1 - th0; rec1 = rec_fn(th1, bufs)
    sh = er.delta_shares(delta, c)
    R["model"].append(mb); R["class"].append(cb); R["steps"].append(out["steps"].cpu()); R["stopped"].append(out["stopped"].cpu())
    if "psi_stop_step" in out: R["psi_stop_step"].append(out["psi_stop_step"].cpu()); R["oracle_step"].append(out["oracle_step"].cpu())
    R["pred0"].append(out["pred0"].cpu()); R["pred1"].append(out["pred1"].cpu()); R["rec0"].append(rec0.cpu()); R["rec1"].append(rec1.cpu())
    for k in ["share_fc_w_c", "share_fc_w_other", "share_fc_b_c", "share_fc_b_other", "share_conv", "sign_b_c", "delta_norm"]: R[k].append(sh[k].cpu())
    R["delta_fc16"].append(delta[:, pidx_store].half().cpu())
    if pidx is None and a.delta_full_n > 0:
        full_sel = np.concatenate([np.where(cb == ci)[0][: a.delta_full_n] for ci in np.unique(cb)])
        R["delta_full16"].append(delta[full_sel].half().cpu()); R["delta_full_model"].append(mb[full_sel])
    if "traj_pred" in out:
        n_log = a.max_steps // a.log_every + 1
        def pad(t):
            t = t.cpu().float(); return torch.cat([t, torch.full((t.shape[0], n_log - t.shape[1]), float("nan"))], 1) if t.shape[1] < n_log else t[:, :n_log]
        R["traj_pred"].append(pad(out["traj_pred"])); traj_step = np.arange(0, a.max_steps + 1, a.log_every)
        R["traj_true"].append(pad(out["traj_true"]) if out["traj_true"] is not None else torch.full((B, n_log), float("nan")))
        R["traj_mon"].append(torch.full((B, n_log), float("nan")))
    if not a.no_derived and kind not in ("ga_real", "ft_retain"):
        add_derived("random_matched", er.random_matched(th0, delta, pidx_store if pidx is None else pidx, gen=gen), th0, bufs, psi)
        if pidx is None: add_derived("random_full_matched", er.random_full_matched(th0, delta, gen=gen), th0, bufs, psi)
        add_derived("bias_matched", er.bias_matched(th0, delta, c, a.task), th0, bufs, psi)
        add_derived("rowscale_matched", er.rowscale_matched(th0, delta, c, a.task), th0, bufs, psi)
        if pidx is None:   # per-layer RV (added 2026-11-12; not yet executed in a re-walk): own generator per batch, drawn AFTER the existing controls so their stream is unchanged
            add_derived("random_layer_matched", er.random_layer_matched(th0, delta, gen=torch.Generator().manual_seed(LAYER_RV_SEED + s)), th0, bufs, psi)
            R.setdefault("layer_group_norm", []).append(er.layer_group_norms(delta).float().cpu())
    for jn, j in judges.items():
        with torch.no_grad(): J[jn]["pred0"].append(j(th0)[ar, c].cpu()); J[jn]["pred1"].append(j(th1)[ar, c].cpu())
    sc = er.scores(rec0, rec1, c, a.task)
    log(f"batch {s//a.batch+1}/{int(np.ceil(len(models)/a.batch))} ({time.time()-t_b:.0f}s): stop {out['stopped'].float().mean():.2f} steps med {out['steps'].float().median():.0f} "
        f"| target {sc['Dt0'].mean():.3f}->{sc['Dt1'].mean():.3f} ctrl Δ {sc['ctrl_change'].mean():+.4f} worst drop med {sc['worst_ctrl_drop'].median():.3f} "
        f"| true-crit {er.criterion_met(sc['Dt1'], sc['Dt0'], a.task).float().mean():.2f} | mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
    del th0, th1, delta, out; torch.cuda.empty_cache()
    if a.time_limit and time.time() - T0 > a.time_limit: log("time limit reached; saving partial"); break

cat = lambda L: (torch.cat(L).numpy() if torch.is_tensor(L[0]) else np.concatenate(L)) if L else np.zeros(0)
save = {k: cat(v) for k, v in R.items() if v}
if traj_step is not None: save["traj_step"] = traj_step
for nm, d in D.items():
    for k, v in d.items(): save[f"{nm}__{k}"] = cat(v)
for jn, d in J.items():
    for k, v in d.items(): save[f"judge__{jn}__{k}"] = cat(v)
save["meta"] = json.dumps({"spec": a.spec, "task": a.task, "set": a.set, "args": vars(a), "seed": SEED, "pidx_kind": ("fc" if pidx is fc_idx else "full"),
                           "n_pidx_store": int(len(pidx_store)), "norm": norm, "maxpool": mp, "time_s": time.time() - T0, "derived": list(D), "root": ROOT, "runs": RUNS, "layer_group_names": er.LAYER_GROUP_NAMES, "layer_rv_seed": LAYER_RV_SEED})
np.savez_compressed(f"results/{tag}.npz", pidx_store=pidx_store.cpu().numpy(), **save)
sc = er.scores(torch.from_numpy(save["rec0"]), torch.from_numpy(save["rec1"]), torch.from_numpy(save["class"]).long(), a.task)
clean = (er.criterion_met(sc["Dt1"], sc["Dt0"], a.task) & (sc["ctrl_change"].abs() <= 0.05)).float().mean()
log(f"DONE n={len(save['model'])} stop_rate {save['stopped'].mean():.3f} true-crit {er.criterion_met(sc['Dt1'], sc['Dt0'], a.task).float().mean():.3f} clean(mean ctrl) {clean:.3f} "
    f"worst-drop<=0.1 {(sc['worst_ctrl_drop'] <= 0.1).float().mean():.3f} total {time.time()-T0:.0f}s")

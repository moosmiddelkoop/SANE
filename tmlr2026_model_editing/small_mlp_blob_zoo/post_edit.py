"""Build per-edit results and figure_data CSVs for the unified blobs unlearning run."""
import os, sys, json, numpy as np, pandas as pd, torch; sys.path.insert(0, os.getcwd())
torch.set_num_threads(int(os.environ.get("NT", "4")))
import ublib as U
from toyzoo import per_class_recall, K
ZOO = "blobs"; SRC = "unified blobs re-run 2026-11-05 (ublib.py/run_walk.py/run_data.py/post_edit.py)"
z = U.load_zoo(); Xe, ye = U.eval_set(); Xp, yp = U.probe_set()
FAM = {"mlp_raw": "MLP", "neuron_r1": "GMN", "sane": "SANE"}
NAME = {"mlp_raw": "paper MLP ψ", "neuron_r1": "GMN ψ (neuron_r1)", "sane": "SANE-W (token head)"}
walks = {}
for tag in ["mlp_raw_full", "mlp_raw_output", "neuron_r1_full", "neuron_r1_output", "sane_full", "sane_output"]:
    f = f"ures/walk_{tag}.npz"
    if os.path.exists(f): walks[tag] = dict(np.load(f))
data = dict(np.load("ures/data_baselines.npz")) if os.path.exists("ures/data_baselines.npz") else None
any_ = next(iter(walks.values())) if walks else data
M, C = any_["model"], any_["cls"]; th0 = torch.tensor(any_["theta0"]); c = torch.tensor(C); B = len(M); ar = np.arange(B)
rec0 = per_class_recall(th0, Xe, ye)[0].numpy()
assert np.abs(rec0 - z["recall"][-1, M]).max() < 1e-6, "eval-set recall must equal stored zoo recall"
states = []   # (method, family, scope, stop_rule, data_based, theta1, steps, fired)
for tag, w in walks.items():
    spec, scope = tag.rsplit("_", 1)
    nm, fam = NAME[spec] + (" (output layer)" if scope == "output" else " (full)"), FAM[spec]
    states += [(nm, fam, scope, "psi", False, w["ms"], w["ms_step"], w["stopped"]),
               (nm, fam, scope, "oracle", False, w["oracle"], w["oracle_step"], None),
               (nm, fam, scope, "fixed", False, w["fixed"], np.full(B, U.FIXED), None)]
if data is not None:
    states += [("GA* (real forget data, data stop)", "GA", "na", "data", True, data["ga"], data["ga_steps"], data["ga_stopped"]),
               ("FT* (real retain data, 100 steps)", "FT", "na", "fixed", True, data["ft"], data["ft_steps"], None)]
if "neuron_r1_full" in walks:
    gen = torch.Generator().manual_seed(11); w = walks["neuron_r1_full"]
    for ref, lab in [("ms", "ψ-stop"), ("oracle", "oracle-stop")]:
        tr = torch.tensor(w[ref])
        states += [(f"RV (norm-matched per layer to GMN-full {lab})", "RV", "na", "matched", False, U.random_matched(th0, tr, gen).numpy(), None, None),
                   (f"Bias (b_c shift norm-matched to GMN-full {lab})", "Bias", "na", "matched", False, U.bias_matched(th0, tr, c).numpy(), None, None)]
rows, ls_rows, rl_rows = [], [], []
oth = np.ones((B, K), bool); oth[ar, C] = False
for (nm, fam, scope, rule, db, th1, steps, fired) in states:
    th1t = torch.tensor(th1); rec1 = per_class_recall(th1t, Xe, ye)[0].numpy()
    t0_, t1_ = rec0[ar, C], rec1[ar, C]; d = rec1 - rec0
    ctrl = (d * oth).sum(1) / (K - 1); worst = np.where(oth, d, np.inf).min(1)
    crit = t1_ <= U.REL * t0_ + U.TOL; clean = crit & (np.abs(ctrl) <= U.CTRL_TOL)
    fs = (fired & ~crit) if fired is not None else np.full(B, np.nan)
    sh = U.block_shares(th1 - th0.numpy(), C)
    rl = U.relearn(th1t, C, Xe=Xe, ye=ye)
    per = pd.DataFrame(dict(zoo=ZOO, method=nm, family=fam, gradient_scope=scope, stop_rule=rule, model=M, cls=C,
                            target_before=t0_, target_after=t1_, ctrl_change=ctrl, worst_ctrl_move=worst, criterion=crit, clean=clean,
                            fired=fired if fired is not None else np.nan, false_stop=fs, steps=steps if steps is not None else np.nan,
                            dnorm_rel=np.linalg.norm(th1 - th0.numpy(), axis=1) / np.linalg.norm(th0.numpy(), axis=1),
                            **{f"share_{k}": v for k, v in sh.items()},
                            **{f"relearn_t_k{k}": v[ar, C].numpy() for k, v in rl.items()},
                            **{f"relearn_ctrl_k{k}": ((v.numpy()) * oth).sum(1) / (K - 1) for k, v in rl.items()}))
    rows.append(per)
    print(f"{nm:58s} {rule:8s} crit {crit.mean():.3f} clean {clean.mean():.3f} ctrl {np.median(ctrl):+.3f} relearn5 {np.median(rl[5][ar, C].numpy()):.3f}", flush=True)
P = pd.concat(rows, ignore_index=True)
os.makedirs("ures/out", exist_ok=True); P.to_parquet("ures/out/blobs_edit_per_edit.parquet")
# ---- faithfulness along metanet paths (editor + second-seed judge of the same family)
fa = []
for tag, w in walks.items():
    spec, scope = tag.rsplit("_", 1)
    nm = NAME[spec] + (" (output layer)" if scope == "output" else " (full)")
    judge = U.get_psi(spec, 1, z, M)
    snaps = np.load(f"ures/snaps_{tag}.npy", mmap_mode="r"); nlog = snaps.shape[0]
    tp, tt = w["traj_psi"], w["traj_true"]; t0p = w["probe_t0"]
    for k in range(nlog):
        step = k * U.LOG
        e_ed = np.abs(tp[:, k] - tt[:, k])
        if judge is not None:
            with torch.no_grad(): pj = judge(torch.tensor(np.asarray(snaps[k])))[ar, C].numpy()
            e_j = np.median(np.abs(pj - tt[:, k]))
        else: e_j = np.nan
        fa.append(dict(zoo=ZOO, method=nm, family=FAM[spec], gradient_scope=scope, step=step, abs_err_editor_median=float(np.median(e_ed)),
                       abs_err_judge_median=float(e_j), frac_psi_stopped=float(((w["ms_step"] <= step) & w["stopped"].astype(bool)).mean()),
                       frac_true_criterion=float((tt[:, k] <= U.REL * t0p).mean()), n=B, source=SRC))
    editor = U.get_psi(spec, 0, z, M)
    for ref, lab in [("ms", "psi_stop"), ("oracle", "oracle_stop")]:
        th = torch.tensor(w[ref]); tr_ = per_class_recall(th, Xp, yp)[0].numpy()[ar, C]
        with torch.no_grad():
            pe = editor(th)[ar, C].numpy(); pj = judge(th)[ar, C].numpy() if judge is not None else np.full(B, np.nan)
        fa.append(dict(zoo=ZOO, method=nm, family=FAM[spec], gradient_scope=scope, step=lab, abs_err_editor_median=float(np.median(np.abs(pe - tr_))),
                       abs_err_judge_median=float(np.median(np.abs(pj - tr_))), frac_psi_stopped=float(w["stopped"].mean()),
                       frac_true_criterion=float((tr_ <= U.REL * t0p).mean()), n=B, source=SRC))
FA = pd.DataFrame(fa)
# ---- aggregate tables
ed, ls, rl = [], [], []
for (nm, fam, scope, rule), g in P.groupby(["method", "family", "gradient_scope", "stop_rule"], sort=False):
    ed.append(dict(zoo=ZOO, method=nm, family=fam, gradient_scope=scope, stop_rule=rule, data_based=fam in ("GA", "FT"), n=len(g),
                   target_before=g.target_before.median(), target_after=g.target_after.median(),
                   target_move_median=(g.target_after - g.target_before).median(), ctrl_change_median=g.ctrl_change.median(),
                   worst_ctrl_move_median=g.worst_ctrl_move.median(), criterion_rate=g.criterion.mean(), clean_success=g.clean.mean(),
                   false_stop_rate=g.false_stop.astype(float).mean() if g.false_stop.notna().any() else np.nan,
                   steps_median=g.steps.median() if g.steps.notna().any() else np.nan, status="final", source=SRC))
    for b in U.BLOCKS:
        gz = g[g.dnorm_rel > 0]   # edits with zero norm (oracle fallback = theta0) carry no layer share
        ls.append(dict(zoo=ZOO, method=nm, family=fam, gradient_scope=scope, stop_rule=rule, block=b, share=gz[f"share_{b}"].mean(), n=len(gz), source=SRC))
    for k in (0, 1, 5, 20):
        rl.append(dict(zoo=ZOO, method=nm, family=fam, gradient_scope=scope, stop_rule=rule, k=k, target_recall_median=g[f"relearn_t_k{k}"].median(),
                       ctrl_recall_median=g[f"relearn_ctrl_k{k}"].median(), n=len(g), source=SRC))
for scope in ("full", "output"):
    if f"sane_{scope}" not in walks:
        nm = NAME["sane"] + (" (output layer)" if scope == "output" else " (full)")
        for rule in ("psi", "oracle", "fixed"):
            ed.append(dict(zoo=ZOO, method=nm, family="SANE", gradient_scope=scope, stop_rule=rule, data_based=False, n=0, status="pending",
                           source="SANE-W walk not finished (run_walk.py sane %s)" % scope))
os.makedirs("figure_data", exist_ok=True)
pd.DataFrame(ed).to_csv("figure_data/blobs_edit.csv", index=False)
pd.DataFrame(ls).to_csv("figure_data/blobs_layer_share.csv", index=False)
pd.DataFrame(rl).to_csv("figure_data/blobs_relearn.csv", index=False)
FA.to_csv("figure_data/blobs_faithfulness.csv", index=False)
print("wrote", len(ed), "edit rows,", len(FA), "faithfulness rows")

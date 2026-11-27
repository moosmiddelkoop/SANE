"""Build editing / faithfulness / manifold / relearn tables from results/*.npz. usage: python analyze_edits.py RESULTS_DIR OUT_PREFIX"""
import os, sys, json, glob, numpy as np, pandas as pd
R, out = sys.argv[1], sys.argv[2]; NC = 100
LABEL = {"meta_gmn_oracle": "graph net ψ, full gradient, ORACLE stop", "meta_fc_gmn_oracle": "graph net ψ, fc-restricted, ORACLE stop", "meta_gmn_fixed100": "graph net ψ, full gradient, fixed 100 steps", "meta_fc_gmn_fixed100": "graph net ψ, fc-restricted, fixed 100 steps", "meta_gmn": "graph net ψ, full gradient (β=0.1)", "meta_gmn_bnorm": "graph net ψ, full gradient (β=0.005)", "meta_fc_gmn": "graph net ψ, fc-restricted gradient",
         "meta_sane_token_last": "SANE-W (token head, last window)", "meta_sane_tokenaug_last": "SANE-W (perturbation-trained token head, last window)",
         "meta_sane_tokenaug_full": "SANE-W full gradient (perturbation-trained head, reduced set)", "meta_sane_mlp_last": "SANE-W (MLP head on z̄, last window)",
         "zdense_sane_token": "SANE-Z dense (token head)", "zdense_sane_tokenaug": "SANE-Z dense (perturbation-trained head)", "meta_fcridge": "ridge-on-fc-stats ψ (weight space)",
         "ga_real": "GA* (256 real class-c images)", "ft_retain": "retain FT* (real images)"}
DER = {"random_matched": "random direction in fc, norm-matched to fc part", "random_full_matched": "random direction (all params), norm-matched", "bias_matched": "fc bias shift, norm-matched", "rowscale_matched": "fc row scaling, norm-matched",
       "identity": "SANE identity (decode(z₀) spliced)", "zrandom": "random walk in z, norm-matched"}


def crit(rec1c, rec0c, task, tol=0.0): return rec1c <= 0.4 * rec0c + tol if task == "unlearn" else rec1c >= 0.9 - tol


def score_rows(rec0, rec1, c, task, stopped=None, steps=None, pred0=None, pred1=None, dnorm=None, extra=None):
    ar = np.arange(len(c)); sign = 1 if task == "unlearn" else -1
    mask = np.ones_like(rec0, bool); mask[ar, c] = False
    Dt0, Dt1 = rec0[ar, c], rec1[ar, c]; ctrl = ((rec1 - rec0) * mask).sum(1) / (NC - 1); worst = np.where(mask, rec0 - rec1, -9).max(1)
    met = crit(Dt1, Dt0, task); met_tol = crit(Dt1, Dt0, task, 0.05)
    row = \
          dict(n=len(c), Dt0=Dt0.mean(), Dt1=Dt1.mean(), Dt1_median=np.median(Dt1), target_move=(sign * (Dt0 - Dt1)).mean(), Dr0=(rec0 * mask).sum(1).mean() / (NC - 1),
               ctrl_change=ctrl.mean(), ctrl_change_median=np.median(ctrl), worst_ctrl_drop_median=np.median(worst), n_ctrl_damaged=float(((rec0 - rec1) > 0.1)[mask].reshape(len(c), -1).sum(1).mean()),
               true_criterion=met.mean(), true_criterion_tol05=met_tol.mean(), clean_success=(met & (np.abs(ctrl) <= 0.05)).mean(),
               clean_success_worst=(met & (worst <= 0.1)).mean(), acc_change=(rec1.mean(1) - rec0.mean(1)).mean())
    if stopped is not None:
        row.update(stop_rate=stopped.mean(), steps_median=float(np.median(steps)), false_stop=(stopped & ~met_tol).sum() / max(stopped.sum(), 1),
                   clean_success_terminated=(met & (np.abs(ctrl) <= 0.05))[stopped].mean() if stopped.any() else np.nan)
    if pred0 is not None and np.isfinite(pred0).any():
        row.update(editor_mae_pre=np.nanmean(np.abs(pred0[ar, c] - Dt0)), editor_mae_post=np.nanmean(np.abs(pred1[ar, c] - Dt1)), psi0=np.nanmean(pred0[ar, c]), psi1=np.nanmean(pred1[ar, c]))
    if dnorm is not None: row.update(delta_norm_median=float(np.median(dnorm)))
    if extra: row.update(extra)
    return row


rows, pcls, traj_rows, judge_rows = [], [], [], []
for f in sorted(glob.glob(os.path.join(R, "*.npz"))):
    b = os.path.basename(f)[:-4]
    if b.startswith("relearn_") or b.startswith("calib_"): continue
    z = np.load(f, allow_pickle=True); m = json.loads(str(z["meta"])); task, spec = m["task"], m["spec"].replace(":", "_")
    key = b.rsplit(f"_{task}_", 1)[0]; label = LABEL.get(key, key); setn = m["set"]
    c = z["class"].astype(int); rec0, rec1 = z["rec0"], z["rec1"]; ar = np.arange(len(c))
    shares = {k: float(np.mean(z[k])) for k in ["share_fc_w_c", "share_fc_w_other", "share_fc_b_c", "share_fc_b_other", "share_conv"] if k in z.files}
    shares["sign_b_c_mean"] = float(np.mean(z["sign_b_c"])) if "sign_b_c" in z.files else np.nan
    r = score_rows(rec0, rec1, c, task, z["stopped"], z["steps"], z["pred0"], z["pred1"], z["delta_norm"], shares)
    r.update(method=label, key=key, task=task, set=setn, kind="metanet" if ("sane" in key or "ridge" in key or "gmn" in key) else "data", pidx=m.get("pidx_kind")); rows.append(r)
    for cc in np.unique(c):
        s = c == cc; rr = score_rows(rec0[s], rec1[s], c[s], task); rr.update(method=label, task=task, cls=int(cc)); pcls.append(rr)
    for d, dl in DER.items():
        if f"{d}__rec1" in z.files:
            r2 = score_rows(rec0, z[f"{d}__rec1"], c, task, z["stopped"], z["steps"], z["pred0"], z[f"{d}__pred1"], z[f"{d}__delta_norm"])
            r2.update(method=f"{dl} [matched to {label}]", key=f"{key}__{d}", task=task, set=setn, kind="control", derived_from=label); rows.append(r2)
    for j in sorted({k.split("__")[1] for k in z.files if k.startswith("judge__")}):
        p0, p1 = z[f"judge__{j}__pred0"], z[f"judge__{j}__pred1"]
        judge_rows.append(dict(method=label, task=task, judge=j, mae_pre=float(np.mean(np.abs(p0 - rec0[ar, c]))), mae_post=float(np.mean(np.abs(p1 - rec1[ar, c]))),
                               judge_pred_move=float(np.mean(p0 - p1)), true_move=float(np.mean(rec0[ar, c] - rec1[ar, c])), corr_move=float(np.corrcoef(p0 - p1, rec0[ar, c] - rec1[ar, c])[0, 1]) if len(c) > 2 else np.nan))
    if "traj_pred" in z.files and z["traj_pred"].ndim == 2 and z["traj_pred"].shape[1] > 1:
        tp, tt, tm, ts = z["traj_pred"], z["traj_true"], z["traj_mon"], z["traj_step"]
        for k, st in enumerate(ts):
            ok = np.isfinite(tp[:, k])
            traj_rows.append(dict(method=label, task=task, step=int(st), n_active=int(ok.sum()), psi_mean=float(np.nanmean(tp[:, k])), true_mean=float(np.nanmean(tt[:, k])) if np.isfinite(tt[:, k]).any() else np.nan,
                                  abs_err_mean=float(np.nanmean(np.abs(tp[:, k] - tt[:, k]))) if np.isfinite(tt[:, k]).any() else np.nan, recon_err_mean=float(np.nanmean(tm[:, k])) if np.isfinite(tm[:, k]).any() else np.nan))
df = pd.DataFrame(rows); first = ["method", "task", "set", "kind", "n", "stop_rate", "steps_median", "Dt0", "Dt1", "target_move", "ctrl_change", "worst_ctrl_drop_median", "n_ctrl_damaged",
                                 "true_criterion", "false_stop", "clean_success", "clean_success_worst", "clean_success_terminated", "editor_mae_pre", "editor_mae_post", "delta_norm_median"]
df = df[[k for k in first if k in df.columns] + [k for k in df.columns if k not in first]]
df.sort_values(["task", "kind", "method"]).to_csv(out + "_editing.csv", index=False)
pd.DataFrame(pcls).to_csv(out + "_editing_per_class.csv", index=False)
pd.DataFrame(traj_rows).to_csv(out + "_trajectories.csv", index=False); pd.DataFrame(judge_rows).to_csv(out + "_judges.csv", index=False)
# relearn
rl = []
for f in sorted(glob.glob(os.path.join(R, "relearn_*.npz"))):
    z = np.load(f, allow_pickle=True); b = os.path.basename(f)[8:-4]; task = "unlearn" if "_unlearn_" in b else "uplearn"; key = b.rsplit(f"_{task}_", 1)[0]
    ks = z["ks"]; c = z["class"].astype(int); rec0 = z["rec0"]
    for mth in sorted({k.split("__")[0] for k in z.files if k.endswith("__idx")}):
        idx = z[f"{mth}__idx"]; cc = c[idx]; ar = np.arange(len(idx)); rk = z[f"{mth}__rec_k"]; r1 = z[f"{mth}__rec1"]; r0 = rec0[idx]
        mask = np.ones_like(r0, bool); mask[ar, cc] = False
        row = dict(method=LABEL.get(key, key) if mth == "main" else f"{DER.get(mth, mth)} [matched to {LABEL.get(key, key)}]", task=task, n=len(idx), target_after_edit=r1[ar, cc].mean(), target_before=r0[ar, cc].mean(),
                   probe_recall_before=float(np.nanmean(z[f"{mth}__probe0"])), probe_recall_after=float(np.nanmean(z[f"{mth}__probe1"])))
        for i, k in enumerate(ks): row[f"target_after_k{k}"] = rk[ar, i, cc].mean(); row[f"ctrl_after_k{k}"] = ((rk[:, i] * mask).sum(1) / (NC - 1)).mean()
        rl.append(row)
if rl: pd.DataFrame(rl).to_csv(out + "_relearn.csv", index=False)
print(df[[k for k in ["method", "task", "n", "stop_rate", "steps_median", "Dt0", "Dt1", "ctrl_change", "worst_ctrl_drop_median", "true_criterion", "false_stop", "clean_success", "clean_success_worst"] if k in df.columns]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print("relearn rows", len(rl), "traj rows", len(traj_rows), "judge rows", len(judge_rows))

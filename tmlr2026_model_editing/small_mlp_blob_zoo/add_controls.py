"""Per-metanet norm-matched RV / Bias controls on the unified blobs edit set (appends to figure_data CSVs).
Same rules and evaluation as post_edit.py; validated by reproducing the existing GMN-full ψ-stop rows."""
import os, sys, numpy as np, pandas as pd, torch; sys.path.insert(0, os.getcwd())
torch.set_num_threads(int(os.environ.get("NT", "4")))
import ublib as U
from toyzoo import per_class_recall, K
ZOO = "blobs"; SRC = "per-metanet norm-matched controls 2026-11-06 (add_controls.py on walk_*.npz from blobs_unified_edit_raw.tar.gz e3ab1346)"
Xe, ye = U.eval_set()
W = {t: dict(np.load(f"ures/walk_{t}.npz")) for t in ["neuron_r1_full", "mlp_raw_full", "mlp_raw_output", "sane_full", "sane_output"]}
M, C = W["mlp_raw_full"]["model"], W["mlp_raw_full"]["cls"]
for w in W.values(): assert np.array_equal(w["model"], M) and np.array_equal(w["cls"], C)
th0 = torch.tensor(W["mlp_raw_full"]["theta0"]); c = torch.tensor(C); B = len(M); ar = np.arange(B)
rec0 = per_class_recall(th0, Xe, ye)[0].numpy(); oth = np.ones((B, K), bool); oth[ar, C] = False

def evaluate(th1):
    th1t = torch.tensor(th1); rec1 = per_class_recall(th1t, Xe, ye)[0].numpy(); t0_, t1_ = rec0[ar, C], rec1[ar, C]; d = rec1 - rec0
    ctrl = (d * oth).sum(1) / (K - 1); worst = np.where(oth, d, np.inf).min(1)
    crit = t1_ <= U.REL * t0_ + U.TOL; clean = crit & (np.abs(ctrl) <= U.CTRL_TOL)
    dn = np.linalg.norm(th1 - th0.numpy(), axis=1)
    return dict(t0=t0_, t1=t1_, ctrl=ctrl, worst=worst, crit=crit, clean=clean, dn=dn, sh=U.block_shares(th1 - th0.numpy(), C),
                rl=U.relearn(th1t, C, Xe=Xe, ye=ye))

def rows_for(nm, fam, scope, ev):
    e = dict(zoo=ZOO, method=nm, family=fam, gradient_scope=scope, stop_rule="matched", data_based=False, n=B,
             target_before=np.median(ev["t0"]), target_after=np.median(ev["t1"]), target_move_median=np.median(ev["t1"] - ev["t0"]),
             ctrl_change_median=np.median(ev["ctrl"]), worst_ctrl_move_median=np.median(ev["worst"]), criterion_rate=ev["crit"].mean(),
             clean_success=ev["clean"].mean(), false_stop_rate=np.nan, steps_median=np.nan, status="final", source=SRC)
    nz = ev["dn"] > 0
    ls = [dict(zoo=ZOO, method=nm, family=fam, gradient_scope=scope, stop_rule="matched", block=b, share=ev["sh"][b][nz].mean(), n=int(nz.sum()), source=SRC) for b in U.BLOCKS]
    rl = [dict(zoo=ZOO, method=nm, family=fam, gradient_scope=scope, stop_rule="matched", k=k, target_recall_median=np.median(ev["rl"][k].numpy()[ar, C]),
               ctrl_recall_median=np.median((ev["rl"][k].numpy() * oth).sum(1) / (K - 1)), n=B, source=SRC) for k in (0, 1, 5, 20)]
    return e, ls, rl

# --- validation: regenerate the existing GMN-full ψ-stop controls (generator seed 11, first draw = ψ-stop RV in post_edit.py)
E_old = pd.read_csv("figure_data/blobs_edit.csv"); R_old = pd.read_csv("figure_data/blobs_relearn.csv"); L_old = pd.read_csv("figure_data/blobs_layer_share.csv")
gen = torch.Generator().manual_seed(11); ref = torch.tensor(W["neuron_r1_full"]["ms"])
chk = {"RV (norm-matched per layer to GMN-full ψ-stop)": U.random_matched(th0, ref, gen).numpy(),
       "Bias (b_c shift norm-matched to GMN-full ψ-stop)": U.bias_matched(th0, ref, c).numpy()}
for nm, th1 in chk.items():
    e, _, _ = rows_for(nm, "x", "na", evaluate(th1)); o = E_old[E_old.method == nm].iloc[0]
    for k in ["criterion_rate", "clean_success", "target_move_median", "ctrl_change_median", "worst_ctrl_move_median"]:
        assert abs(e[k] - o[k]) < 1e-7, (nm, k, e[k], o[k])
print("validation: existing GMN-full ψ-stop RV/Bias rows reproduced exactly", flush=True)

# --- new parents
PARENT = {"mlp_raw_full": ("paper-MLP-full", "full"), "mlp_raw_output": ("paper-MLP-output", "output"),
          "sane_full": ("SANE-W-full", "full"), "sane_output": ("SANE-W-output", "output")}
ed, ls, rl = [], [], []
for i, (tag, (pname, scope)) in enumerate(PARENT.items()):
    gen = torch.Generator().manual_seed(100 + i)          # one fixed generator per parent; ψ-stop drawn first, then oracle-stop
    for ref, lab in [("ms", "ψ-stop"), ("oracle", "oracle-stop")]:
        tr = torch.tensor(W[tag][ref])
        for fam, nm, th1 in [("RV", f"RV (norm-matched per layer to {pname} {lab})", U.random_matched(th0, tr, gen).numpy()),
                             ("Bias", f"Bias (b_c shift norm-matched to {pname} {lab})", U.bias_matched(th0, tr, c).numpy())]:
            e, l, r = rows_for(nm, fam, scope, evaluate(th1)); ed.append(e); ls += l; rl += r
            print(f"{nm:62s} crit {e['criterion_rate']:.4f} clean {e['clean_success']:.4f} move {e['target_move_median']:+.3f} "
                  f"ctrl {e['ctrl_change_median']:+.3f} worst {e['worst_ctrl_move_median']:+.3f} zero-norm {int((~(np.linalg.norm(th1 - th0.numpy(), axis=1) > 0)).sum())}", flush=True)
for old, new, f in [(E_old, ed, "blobs_edit.csv"), (R_old, rl, "blobs_relearn.csv"), (L_old, ls, "blobs_layer_share.csv")]:
    N = pd.DataFrame(new)[old.columns]; assert not set(N.method) & set(old.method)
    pd.concat([old, N], ignore_index=True).to_csv(f"figure_data/{f}", index=False); print(f, len(old), "->", len(old) + len(N))

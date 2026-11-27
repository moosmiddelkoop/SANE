"""Re-derive per-edit records of add_controls.py (ab3d2fe1) controls; same generators/draw order, relearn/shares skipped (no RNG coupling)."""
import os, sys, numpy as np, pandas as pd, torch; sys.path.insert(0, os.getcwd()); torch.set_num_threads(8)
import ublib as U
from toyzoo import per_class_recall, K
Xe, ye = U.eval_set()
W = {t: dict(np.load(f"ures/walk_{t}.npz")) for t in ["neuron_r1_full", "mlp_raw_full", "mlp_raw_output", "sane_full", "sane_output"]}
M, C = W["mlp_raw_full"]["model"], W["mlp_raw_full"]["cls"]
for w in W.values(): assert np.array_equal(w["model"], M) and np.array_equal(w["cls"], C)
th0 = torch.tensor(W["mlp_raw_full"]["theta0"]); c = torch.tensor(C); B = len(M); ar = np.arange(B)
rec0 = per_class_recall(th0, Xe, ye)[0].numpy(); oth = np.ones((B, K), bool); oth[ar, C] = False
def ev(th1):
    rec1 = per_class_recall(torch.tensor(th1), Xe, ye)[0].numpy(); d = rec1 - rec0
    return pd.DataFrame(dict(model=M, cls=C, target_before=rec0[ar, C], target_after=rec1[ar, C],
        ctrl_change=(d * oth).sum(1) / (K - 1), worst_ctrl_move=np.where(oth, d, np.inf).min(1)))
out = []
# validation: GMN-full psi-stop AND oracle-stop controls (post_edit.py: gen seed 11, psi drawn first then oracle)
gen = torch.Generator().manual_seed(11)
for ref, lab in [("ms", "ψ-stop"), ("oracle", "oracle-stop")]:
    tr = torch.tensor(W["neuron_r1_full"][ref])
    for nm, th1 in [(f"RV (norm-matched per layer to GMN-full {lab})", U.random_matched(th0, tr, gen).numpy()),
                    (f"Bias (b_c shift norm-matched to GMN-full {lab})", U.bias_matched(th0, tr, c).numpy())]:
        out.append(ev(th1).assign(method=nm, stop_rule="matched", role="validation"))
PARENT = {"mlp_raw_full": "paper-MLP-full", "mlp_raw_output": "paper-MLP-output", "sane_full": "SANE-W-full", "sane_output": "SANE-W-output"}
for i, (tag, pname) in enumerate(PARENT.items()):
    gen = torch.Generator().manual_seed(100 + i)
    for ref, lab in [("ms", "ψ-stop"), ("oracle", "oracle-stop")]:
        tr = torch.tensor(W[tag][ref])
        for nm, th1 in [(f"RV (norm-matched per layer to {pname} {lab})", U.random_matched(th0, tr, gen).numpy()),
                        (f"Bias (b_c shift norm-matched to {pname} {lab})", U.bias_matched(th0, tr, c).numpy())]:
            out.append(ev(th1).assign(method=nm, stop_rule="matched", role="regenerated"))
P = pd.concat(out, ignore_index=True); P.to_parquet("ures/out/blobs_matched_controls_per_edit_regen.parquet"); print(P.groupby(["role","method"]).size().to_string())

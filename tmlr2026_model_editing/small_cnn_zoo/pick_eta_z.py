"""Choose eta_z for SANE-Z from validation calibration runs results/calib_<task>_<eta>.npz.
Criterion: maximise mean clean success (target move >= 0.2 and |mean control change| <= 0.05) over both tasks,
subject to the median |control change| not exceeding the sane_identity control's by more than 0.02 (if present).
usage: python pick_eta_z.py eta1 eta2 ... -> prints eta, writes eta_z.txt and results/calib_summary.json"""
import sys, json, glob, numpy as np
# usage note: pass the eta tags exactly as used in the calibration file names (e.g. 1e-3 3e-3 1e-2 3e-2);
# earlier versions formatted the tag as f"{float:g}" (0.001), which never matched calib_<task>_1e-3.npz
# and silently returned empty metrics (all scores -1, choice = first grid value). Fixed 2026-10-09.
tags = sys.argv[1:]
summary = {}
for tag in tags:
    e = float(tag)
    row = {}
    for task in ("unlearn", "uplearn"):
        fs = sorted(set(glob.glob(f"results/calib_{task}_{tag}.npz") + glob.glob(f"results/calib_{task}_{e:g}.npz")))
        if not fs:
            raise SystemExit(f"missing calibration file for task={task} eta={tag}")
        if not fs:
            continue
        z = np.load(fs[0])
        for m in sorted({k.split("__")[0] for k in z.files if "__" in k}):
            tm, cc = z[f"{m}__target_move"], z[f"{m}__ctrl_change"]
            row[f"{task}_{m}_clean"] = float(np.mean((tm >= 0.2) & (np.abs(cc) <= 0.05)))
            row[f"{task}_{m}_move"] = float(np.mean(tm)); row[f"{task}_{m}_ctrl_med"] = float(np.median(np.abs(cc)))
            row[f"{task}_{m}_n"] = int(len(tm))
    summary[f"{e:g}"] = row
def score(row):
    cl = [v for k, v in row.items() if k.endswith("_clean") and ("_zedit_" in k)]
    return float(np.mean(cl)) if cl else -1.0
ok = {}
for e, row in summary.items():
    pen = 0.0
    for task in ("unlearn", "uplearn"):
        a, b = row.get(f"{task}_zedit_ctrl_med"), row.get(f"{task}_sane_identity_ctrl_med")
        if a is not None and b is not None and a > b + 0.02:
            pen += 1
    ok[e] = (pen == 0, score(row))
cands = [e for e, (p, s) in ok.items() if p] or list(summary)
best = max(cands, key=lambda e: ok[e][1])
summary["_choice"] = {"eta_z": float(best), "scores": {e: ok[e][1] for e in summary if not e.startswith("_")}, "constraint_ok": {e: ok[e][0] for e in ok}}
json.dump(summary, open("results/calib_summary.json", "w"), indent=1)
open("eta_z.txt", "w").write(f"{float(best):g}\n"); print(f"{float(best):g}")

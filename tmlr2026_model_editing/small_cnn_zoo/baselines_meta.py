"""Prediction baselines that need no metanet: mean-recall oracle and ridge on permutation-invariant features.
Writes runs/baselines.json and runs/ridge_pred_test.npy (same row order as run_meta's test set: all 3 checkpoints, split=='test').
usage: python baselines_meta.py [--alpha 10]"""
import os, sys, json, time, argparse
os.environ["KMP_AFFINITY"] = "disabled"; os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.append(os.getcwd())
import numpy as np, torch
import cnnzoo, meta_cnn as mc

ap = argparse.ArgumentParser(); ap.add_argument("--alpha", type=float, default=10.0); args = ap.parse_args()
os.makedirs("runs", exist_ok=True); t0 = time.time()
meta = np.load("zoo_meta.npz", allow_pickle=False)
TH = np.load("zoo_theta.npy", mmap_mode="r"); M = len(meta["model"])
steps_all = np.repeat(np.array([0, 20, 86]), M)
Y = meta["recall"].reshape(3 * M, 10).astype(np.float64); acc_all = meta["acc"].reshape(3 * M)
act_all = np.tile(meta["act"].astype(np.int64), 3); split_all = np.tile(meta["split"], 3)
tr = np.where(split_all == "train")[0]; te = np.where(split_all == "test")[0]; va = np.where(split_all == "val")[0]


def feats(idx, chunk=4096):
    out = []
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        out.append(mc.invariant_features(torch.as_tensor(np.ascontiguousarray(TH[ii])), torch.as_tensor(act_all[ii])).numpy())
    return np.concatenate(out).astype(np.float64)


Ftr, Fva, Fte = feats(tr), feats(va), feats(te)
mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-8
Ftr, Fva, Fte = (Ftr - mu) / sd, (Fva - mu) / sd, (Fte - mu) / sd
X = np.hstack([Ftr, np.ones((len(tr), 1))])
A = X.T @ X + args.alpha * np.eye(X.shape[1]); A[-1, -1] -= args.alpha
W = np.linalg.solve(A, X.T @ Y[tr])
pred = lambda F: np.clip(np.hstack([F, np.ones((len(F), 1))]) @ W, 0, 1)
Pte, Pva = pred(Fte), pred(Fva)
res = {"alpha": args.alpha, "n_features": int(Ftr.shape[1]), "val_r2_all": mc.r2(Y[va], Pva)}
res.update(mc.metric_table(Y[te], Pte, steps_all[te], acc_all[te], act_all[te], prefix="ridge_test_"))
# mean-recall oracle: predicts each model's own mean recall for every class
O = np.repeat(Y[te].mean(1, keepdims=True), 10, 1)
res.update(mc.metric_table(Y[te], O, steps_all[te], acc_all[te], act_all[te], prefix="oracle_test_"))
# zoo-mean predictor (constant per class, from train split) as the floor
C = np.repeat(Y[tr].mean(0, keepdims=True), len(te), 0)
res.update(mc.metric_table(Y[te], C, steps_all[te], acc_all[te], act_all[te], prefix="const_test_"))
res["time_s"] = time.time() - t0
json.dump(res, open("runs/baselines.json", "w"), indent=1)
np.save("runs/ridge_pred_test.npy", Pte.astype(np.float32))
print(f"ridge r2_all {res['ridge_test_r2_all']:.4f} final {res['ridge_test_r2_final']:.4f} centred {res['ridge_test_r2_centred_final_trained']:.4f} | "
      f"oracle r2_all {res['oracle_test_r2_all']:.4f} final {res['oracle_test_r2_final']:.4f} | ({res['time_s']:.0f}s)")

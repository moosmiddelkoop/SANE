"""Stage one epoch of the CIFAR-100 ResNet-18 zoo (modelzoos.cc layout, extracted epochs) into the train_gmn.py layout:
zoo_theta.f32 (M,P) fp32 memmap, zoo_bn.npy (M,NB), zoo_meta.npz (recall/test_acc/... at that epoch, split copied from the
epoch-60 staged zoo by model dir), zoo_std.npz (per-tensor mu/sigma over the train split), cifar100.npz symlink. No tokens.
usage: python stage_zoo_epoch.py ZOO_DIR EPOCH OUT_DIR REF_META CIFAR_NPZ [--workers 8] [--limit N]"""
import os, sys, json, time, argparse, re
import numpy as np, torch
from multiprocessing import Pool
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import resnet_zoo as rz
torch.set_num_threads(2)
ap = argparse.ArgumentParser(); ap.add_argument("zoo"); ap.add_argument("epoch", type=int); ap.add_argument("out"); ap.add_argument("ref_meta"); ap.add_argument("cifar")
ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T0 = time.time(); E = a.epoch
log = lambda *x: print(f"[ep{E} {time.time()-T0:6.0f}s]", *x, flush=True)

dirs = sorted([d for d in os.listdir(a.zoo) if d.startswith("NN_tune_trainable")], key=lambda d: int(re.search(r"_(\d{5})_", d).group(1)))
if a.limit: dirs = dirs[:a.limit]
M = len(dirs); log("models", M)
# split: copied from the epoch-60 staged zoo (same 70/15/15 split_id), matched by model dir name, fallback by 5-digit model index
ref = np.load(a.ref_meta, allow_pickle=True)
ref_split = {str(d): str(s) for d, s in zip(ref["model_dir"], ref["split"])}
ref_by_idx = {re.search(r"_(\d{5})_", str(d)).group(1): str(s) for d, s in zip(ref["model_dir"], ref["split"])}
n_name, n_idx = 0, 0
def split_of(d):
    global n_name, n_idx
    if d in ref_split: n_name += 1; return ref_split[d]
    k = re.search(r"_(\d{5})_", d).group(1); n_idx += 1; return ref_by_idx.get(k, "none")
splits = np.array([split_of(d) for d in dirs]); log("split matched by name", n_name, "by index", n_idx, {s: int((splits == s).sum()) for s in np.unique(splits)})


def load_one(d):
    sd = torch.load(os.path.join(a.zoo, d, f"checkpoint_{E:06d}", "checkpoints"), map_location="cpu", weights_only=False)
    th, buf = rz.sd_to_flat(sd)
    lines = [json.loads(l) for l in open(os.path.join(a.zoo, d, "result.json")) if l.strip()]
    res = [r for r in lines if r["training_iteration"] == E]
    assert len(res) == 1, (d, E, [r["training_iteration"] for r in lines][:5])
    res = res[0]
    rec = np.array([res[f"acc_class_{k}"] for k in range(rz.NC)], dtype=np.float32)
    return th, buf, rec, float(res["test_acc"]), float(res["val_acc"]), float(res["train_acc"]), int(res["config"]["seed"])


theta = np.memmap(os.path.join(a.out, "zoo_theta.f32"), dtype=np.float32, mode="w+", shape=(M, rz.P))
bn = np.zeros((M, rz.NB), np.float32); recall = np.zeros((M, rz.NC), np.float32)
test_acc = np.zeros(M, np.float32); val_acc = np.zeros(M, np.float32); train_acc = np.zeros(M, np.float32); seed = np.zeros(M, np.int64)
def _fill(i, r):
    th, buf, rec, ta, va, tra, sd_ = r
    theta[i] = th; bn[i] = buf; recall[i] = rec; test_acc[i] = ta; val_acc[i] = va; train_acc[i] = tra; seed[i] = sd_
    if i % 100 == 0: log(f"loaded {i}/{M}")
if a.workers > 1:
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap(load_one, dirs, chunksize=4)): _fill(i, r)
else:
    for i, d in enumerate(dirs): _fill(i, load_one(d))
theta.flush()
np.save(os.path.join(a.out, "zoo_bn.npy"), bn)
np.savez(os.path.join(a.out, "zoo_meta.npz"), recall=recall, test_acc=test_acc, val_acc=val_acc, train_acc=train_acc, seed=seed,
         split=splits, model_dir=np.array(dirs), n_models=M, split_id=str(ref["split_id"]), epoch=E)
log("meta saved; acc min/med/max", test_acc.min(), np.median(test_acc), test_acc.max(), "recall mean", recall.mean(), "bn var min", bn.min())

# per-tensor standardisation statistics over the train split (mu = mean of per-model means, sigma = RMS of per-model stds)
tr = np.where(splits == "train")[0]
means = np.zeros((len(tr), rz.N_TENSORS)); stds = np.zeros((len(tr), rz.N_TENSORS))
for j, i in enumerate(tr):
    th = torch.from_numpy(np.array(theta[i]))
    for t, (k, s) in enumerate(rz.PARAM_SPECS):
        a_, b_, _ = rz.OFFSETS[k]; w = th[a_:b_]
        means[j, t] = w.mean().item(); stds[j, t] = w.std().item()
mu = means.mean(0).astype(np.float32); sigma = np.sqrt((stds ** 2).mean(0)).astype(np.float32)
np.savez(os.path.join(a.out, "zoo_std.npz"), mu=mu, sigma=sigma, keys=np.array([k for k, _ in rz.PARAM_SPECS]))
dst = os.path.join(a.out, "cifar100.npz")
if not os.path.exists(dst): os.symlink(os.path.abspath(a.cifar), dst)
# eligible unlearning edits per class on the test split (recall_c >= 0.5)
te = np.where(splits == "test")[0]; elig = (recall[te] >= 0.5).sum(0)
json.dump({"n_models": M, "epoch": E, "P": rz.P, "time_s": time.time() - T0, "split_counts": {s: int((splits == s).sum()) for s in np.unique(splits)},
           "test_acc_mean": float(test_acc.mean()), "test_acc_sd": float(test_acc.std()), "recall_mean": float(recall.mean()),
           "unlearn_eligible_test_per_class_min_med_max": [int(elig.min()), float(np.median(elig)), int(elig.max())],
           "classes_with_lt6_eligible": int((elig < 6).sum()), "classes_with_0_eligible": int((elig == 0).sum())},
          open(os.path.join(a.out, "stage_info.json"), "w"), indent=1)
log("DONE eligible/class min/med/max", elig.min(), np.median(elig), elig.max(), "classes <6:", (elig < 6).sum())

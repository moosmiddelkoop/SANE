"""Worker: train one metanet config on the MNIST zoo and write runs/<spec>_s<seed>_e<epochs>.{json,pt,pred_test.npy}
usage: python run_meta.py SPEC SEED EPOCHS [--small N] [--time_limit S]"""
import os, sys, json, time, argparse
os.environ["KMP_AFFINITY"] = "disabled"
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import cnnzoo, meta_cnn as mc

ap = argparse.ArgumentParser()
ap.add_argument("spec"); ap.add_argument("seed", type=int); ap.add_argument("epochs", type=int)
ap.add_argument("--small", type=int, default=0, help="subsample train models (debug)")
ap.add_argument("--time_limit", type=float, default=None)
ap.add_argument("--steps", default="0,20,86", help="which checkpoints to train on")
ap.add_argument("--tag", default="")
args = ap.parse_args()

os.makedirs("runs", exist_ok=True)
name = f"{args.spec}_s{args.seed}_e{args.epochs}{args.tag}"
if args.small:
    name += f"_small{args.small}"
if os.path.exists(f"runs/{name}.json") and not os.environ.get("FORCE_RERUN"):
    print(f"[{name}] exists, skipping"); raise SystemExit(0)

# ---------------- data
meta = np.load("zoo_meta.npz", allow_pickle=False)
TH = np.load("zoo_theta.npy", mmap_mode="r")          # (3*M, 4970) rows ordered step-major (0,20,86)
M = len(meta["model"])
steps_all = np.repeat(np.array([0, 20, 86]), M)
recall_all = meta["recall"].reshape(3 * M, 10).astype(np.float32)
acc_all = meta["acc"].reshape(3 * M)
act_all = np.tile(meta["act"].astype(np.int64), 3)
split_all = np.tile(meta["split"], 3)
use_steps = [int(s) for s in args.steps.split(",")]
in_steps = np.isin(steps_all, use_steps)

rng = np.random.default_rng(args.seed)
tr = np.where((split_all == "train") & in_steps)[0]
if args.small:
    tr = np.sort(rng.choice(tr, min(args.small, len(tr)), replace=False))
va = np.where((split_all == "val") & in_steps)[0]
te = np.where(split_all == "test")[0]
if args.small:
    va = np.sort(rng.choice(va, min(args.small // 2, len(va)), replace=False))


def rows(idx):  # materialise rows from memmap (sorted idx is fastest)
    return np.ascontiguousarray(TH[idx])


t0 = time.time()
Xtr = rows(tr); Xva = rows(va)
print(f"[{name}] loaded train {Xtr.shape} val {Xva.shape} in {time.time()-t0:.0f}s", flush=True)
Ytr, Yva = recall_all[tr], recall_all[va]
Atr, Ava = act_all[tr], act_all[va]

# ---------------- model
spec = args.spec
aug = ()
torch.manual_seed(args.seed)
if spec == "mlp_paper":
    net = mc.MLPMeta(use_act=False)
elif spec == "mlp_act":
    net = mc.MLPMeta(use_act=True)
elif spec in ("mlp_aug", "mlp_std"):
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    net = mc.MLPMeta(use_act=True, mu=mu, sd=sd)
    if spec == "mlp_aug":
        aug = (mc.aug_permute, mc.aug_signflip)
elif spec.startswith("graph"):
    rounds = int(spec[5:])
    sidx = np.sort(np.random.default_rng(0).choice(len(Xtr), min(4096, len(Xtr)), replace=False))   # all checkpoints, not only step 0
    p = cnnzoo.unflatten(torch.as_tensor(Xtr[sidx]))
    scale = {k: max(float(v.pow(2).mean().sqrt()), 1e-2) for k, v in p.items()}
    net = mc.ChannelGraphNet(d=64, rounds=rounds, scale=scale)
elif spec.startswith("probe_"):
    _, mode, P = spec.split("_"); P = int(P)
    init_images = None
    if mode in ("real", "fixed"):
        Xtr_img, ytr_img, _, _ = cnnzoo.load_images(os.environ.get("ZOO", "mnist"))       # train images only
        g = np.random.default_rng(1000 + args.seed)
        # class-balanced pick
        per = P // 10; pick = []
        for c in range(10):
            pick += list(g.choice(np.where(ytr_img == c)[0], per, replace=False))
        pick += list(g.choice(len(ytr_img), P - len(pick), replace=False))
        init_images = cnnzoo.prep_images(Xtr_img[np.array(pick)]).numpy()
    net = mc.ProbeNet(P=P, mode=mode, seed=args.seed, init_images=init_images)
else:
    raise SystemExit(f"unknown spec {spec}")
n_par = sum(p.numel() for p in net.parameters())
print(f"[{name}] params {n_par}", flush=True)

net, best, hist = mc.train(net, Xtr, Atr, Ytr, Xva, Ava, Yva, epochs=args.epochs, seed=args.seed, aug=aug,
                           time_limit=args.time_limit, log=f"runs/{name}_hist.json", clip=1.0 if spec.startswith("graph") else None)

# ---------------- evaluate on full test (all 3 checkpoints) + val
Xte = rows(te)
Pte = mc.predict(net, Xte, act_all[te])
Pva = mc.predict(net, Xva, Ava)
res = {"spec": spec, "seed": args.seed, "epochs": args.epochs, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
       "params": n_par, "best_val_r2": best, "train_time_s": hist[-1]["time"], "steps_trained": use_steps}
res.update(mc.metric_table(recall_all[te], Pte, steps_all[te], acc_all[te], act_all[te], prefix="test_"))
res["val_r2_all"] = mc.r2(Yva, Pva)
res["hist"] = hist
json.dump(res, open(f"runs/{name}.json", "w"), indent=1)
np.save(f"runs/{name}_pred_test.npy", Pte.astype(np.float32))
torch.save({"spec": spec, "seed": args.seed, "state_dict": net.state_dict(),
            "init": getattr(net, "mode", None), "P": getattr(net, "P", None)}, f"runs/{name}.pt")
print(f"[{name}] DONE test r2_all {res['test_r2_all']:.4f} r2_final {res['test_r2_final']:.4f} "
      f"r2_centred {res['test_r2_centred_final_trained']:.4f} ({time.time()-t0:.0f}s)", flush=True)

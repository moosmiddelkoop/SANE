"""Token-space reconstruction of a trained windowed SANE AE on val and test models (same masked-recon loss / R^2 as run_pretrain.eval_val,
non-overlapping 512-token windows at offsets 0, W, 2W, ...; the last partial window is excluded exactly as in eval_val).
usage: python eval_recon_test.py ZOO_ROOT SANE_PT OUT_JSON"""
import os, sys, json, time
sys.path.append(os.getcwd())
import numpy as np, torch
import resnet_zoo as rz, sane_resnet as sr
from sane_vendor.def_loss import GammaContrastReconLoss
root, pt, outp = sys.argv[1:4]; dev = rz.DEV; T0 = time.time()
zoo = rz.Zoo(root); sane, _, ck = sr.load_sane(pt, dev); sane.eval(); W = sane.W
a = ck["meta"]["args"] if isinstance(ck, dict) and "meta" in ck else {}
crit = GammaContrastReconLoss(gamma=0.05, reduction="mean", batch_size=64, temperature=0.1).to(dev)
MASK, POS = sane.mask, sane.pos
res = {"sane_pt": pt, "window": W, "n_windows_per_model_scored": len(range(0, rz.N_TOK - W + 1, W))}
for split in ("val", "test", "train"):
    ids = zoo.idx(split)
    if split == "train": ids = ids[:150]
    per_model = []; tot = rsq = cnt = 0
    with torch.no_grad():
        for m in ids:
            x_all = torch.from_numpy(np.array(zoo.tokens[m])).to(dev).float(); ml = []
            for s in range(0, rz.N_TOK - W + 1, W):
                x = x_all[s:s + W][None]; p = POS[s:s + W][None]; mk = MASK[s:s + W][None].float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    z, y, _ = sane(x, p)
                o = crit.loss_recon(y.float(), x, mk); tot += o["loss_recon"].item(); rsq += float(o["rsq"]); cnt += 1; ml.append(float(o["rsq"]))
            per_model.append(float(np.mean(ml)))
    res[split] = {"n_models": int(len(ids)), "recon_loss_mean": tot / cnt, "rsq_mean_over_windows": rsq / cnt,
                  "rsq_per_model_median": float(np.median(per_model)), "rsq_per_model_q05_q95": [float(np.percentile(per_model, 5)), float(np.percentile(per_model, 95))]}
    print(split, res[split], f"{time.time()-T0:.0f}s", flush=True)
res["note"] = "train row uses the first 150 train models (in-sample reference)"
json.dump(res, open(outp, "w"), indent=1); print("DONE")

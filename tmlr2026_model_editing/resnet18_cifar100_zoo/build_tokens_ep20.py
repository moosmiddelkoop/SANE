"""Build the SANE token view of an already-staged epoch zoo (stage_zoo_epoch.py layout) without touching the source dir.
usage: python build_tokens_ep20.py SRC_ROOT OUT_ROOT
OUT_ROOT gets symlinks to SRC_ROOT/{zoo_theta.f32, zoo_bn.npy, zoo_meta.npz, zoo_std.npz, cifar100.npz, edit_sets.npz} and a new
zoo_tokens.f16 (M, N_TOK, D_T) = tokenize((theta - mu_p) / sigma_p) in fp16, exactly as stage_zoo.py of the SANE ResNet study
(mu/sigma = per-tensor train-split statistics in zoo_std.npz). Written to a .tmp file and renamed when complete; skipped if present."""
import os, sys, json, time
sys.path.append(os.getcwd())
import numpy as np, torch
import resnet_zoo as rz
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True); T0 = time.time()
log = lambda *x: print(f"[tokens {time.time()-T0:6.0f}s]", *x, flush=True)
for f in ["zoo_theta.f32", "zoo_bn.npy", "zoo_meta.npz", "zoo_std.npz", "cifar100.npz", "edit_sets.npz"]:
    d = os.path.join(out, f)
    if not os.path.lexists(d) and os.path.exists(os.path.join(src, f)): os.symlink(os.path.realpath(os.path.join(src, f)), d)
tp = os.path.join(out, "zoo_tokens.f16")
if os.path.exists(tp): log("tokens exist, skipping build"); raise SystemExit(0)
zoo = rz.Zoo(out); M = zoo.M
st = np.load(os.path.join(out, "zoo_std.npz")); mu = st["mu"]; sigma = st["sigma"]
mu_p = torch.from_numpy(mu)[torch.from_numpy(rz.TENSOR_ID)]; sg_p = torch.from_numpy(sigma)[torch.from_numpy(rz.TENSOR_ID)]
tmp = tp + ".tmp"; mx = 0.0; nbytes = rz.N_TOK * rz.D_T * 2
with open(tmp, "wb") as fh:
    for i in range(M):
        th = (torch.from_numpy(np.array(zoo.theta[i])) - mu_p) / sg_p
        tok = rz.tokenize(th[None])[0]
        mx = max(mx, tok.abs().max().item()); b = tok.numpy().astype(np.float16).tobytes(); assert len(b) == nbytes; fh.write(b)
        if i % 100 == 0:
            fh.flush(); os.fsync(fh.fileno())
            if i == 0:   # round-trip check on model 0 (fp16 tolerance)
                back = rz.untokenize(torch.from_numpy(tok.numpy().astype(np.float16)).float()[None])[0]
                log("round-trip max |diff| (std units, fp16)", float((back - th).abs().max()))
            log(f"tokens {i}/{M} max|tok| so far {mx:.1f}")
    fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, tp)
json.dump({"src": src, "M": M, "N_TOK": rz.N_TOK, "D_T": rz.D_T, "max_abs_token": mx, "time_s": time.time() - T0,
           "split_counts": {s: int((zoo.split == s).sum()) for s in np.unique(zoo.split)}}, open(os.path.join(out, "tokens_info.json"), "w"), indent=1)
log("DONE max|tok|", mx)

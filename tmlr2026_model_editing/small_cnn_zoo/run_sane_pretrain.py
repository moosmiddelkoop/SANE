"""SANE pretraining on a Small-CNN-Zoo + property heads + identity (reconstruction) control.
usage: python run_sane_pretrain.py SEED EPOCHS [--no_align] [--small N] [--batch 64] [--lr 1e-4] [--d_model 512]
                                   [--layers 4] [--lat 128] [--nhead 8] [--gamma 0.05] [--noise 0.1] [--tag T]
                                   [--time_limit S] [--identity_n 200] [--restarts 8]
Reads zoo_meta.npz / zoo_theta.npy (prep_zoo.py layout) and the zoo's image file (ZOO env var) in the cwd.
Writes runs/sane<tag>_s<seed>_e<epochs>.pt (AE + heads), runs/sane_<head>_s<seed>_e<epochs>{tag}.{json,pred_test.npy}
(one per head, same format as run_meta.py so the prediction tables pick them up),
runs/sane<tag>_s<seed>_e<epochs>.json (pretraining metrics) and runs/sane_perms.npy (shared alignment cache)."""
import os, sys, json, time, argparse
os.environ["KMP_AFFINITY"] = "disabled"
os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.append(os.getcwd())
import numpy as np, torch, torch.nn as nn
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
import cnnzoo, meta_cnn as mc, sane_zoo as sz
from sane_vendor.def_loss import GammaContrastReconLoss

ap = argparse.ArgumentParser()
ap.add_argument("seed", type=int); ap.add_argument("epochs", type=int)
ap.add_argument("--no_align", action="store_true"); ap.add_argument("--small", type=int, default=0)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--wd", type=float, default=3e-9)
ap.add_argument("--d_model", type=int, default=512); ap.add_argument("--layers", type=int, default=4)
ap.add_argument("--lat", type=int, default=128); ap.add_argument("--nhead", type=int, default=8)
ap.add_argument("--gamma", type=float, default=0.05); ap.add_argument("--temperature", type=float, default=0.1)
ap.add_argument("--noise", type=float, default=0.1); ap.add_argument("--tag", default="")
ap.add_argument("--time_limit", type=float, default=None); ap.add_argument("--identity_n", type=int, default=200)
ap.add_argument("--restarts", type=int, default=8); ap.add_argument("--head_epochs", type=int, default=60)
ap.add_argument("--perms_only", action="store_true", help="compute/cache the alignment permutations and exit")
args = ap.parse_args()

DEV = cnnzoo.DEV
ZOO = os.environ.get("ZOO", "mnist")
os.makedirs("runs", exist_ok=True)
tag = args.tag + ("_noalign" if args.no_align else "")
name = f"sane{tag}_s{args.seed}_e{args.epochs}" + (f"_small{args.small}" if args.small else "")
if os.path.exists(f"runs/{name}.json") and not os.environ.get("FORCE_RERUN") and not args.perms_only:
    print(f"[{name}] exists, skipping"); raise SystemExit(0)
T0 = time.time()
log = lambda *a: print(f"[{name} {time.time()-T0:6.0f}s]", *a, flush=True)

# ---------------- data (same conventions as run_meta.py)
meta = np.load("zoo_meta.npz", allow_pickle=False)
TH = np.load("zoo_theta.npy", mmap_mode="r")
M = len(meta["model"])
steps_all = np.repeat(np.array([0, 20, 86]), M)
recall_all = meta["recall"].reshape(3 * M, 10).astype(np.float32)
acc_all = meta["acc"].reshape(3 * M); act_all = np.tile(meta["act"].astype(np.int64), 3)
split_all = np.tile(meta["split"], 3)
rng = np.random.default_rng(args.seed)
tr = np.where(split_all == "train")[0]; va = np.where(split_all == "val")[0]; te = np.where(split_all == "test")[0]
if args.small:
    tr = np.sort(rng.choice(tr, min(args.small, len(tr)), replace=False))
    va = np.sort(rng.choice(va, min(args.small // 2, len(va)), replace=False))
    te = np.sort(rng.choice(te, min(args.small // 2, len(te)), replace=False))
rows = lambda idx: torch.from_numpy(np.ascontiguousarray(TH[idx]))

# ---------------- alignment (shared cache; computed on epoch-86 rows, reused for epochs 0/20)
train_models = np.where(meta["split"] == "train")[0]
ref_idx = sz.choose_reference(meta["acc"][2][train_models], train_models)
theta_ref = rows(np.array([2 * M + ref_idx]))[0]
perm_file = "runs/sane_perms.npy"
if args.no_align:
    PERMS = sz.identity_perms(M)
elif os.path.exists(perm_file):
    PERMS = torch.from_numpy(np.load(perm_file)).long(); log("loaded perms", perm_file)
else:
    t0 = time.time(); PERMS = torch.zeros(M, 3, 16, dtype=torch.long); OBJ = torch.zeros(M); OBJ0 = torch.zeros(M)
    for s in range(0, M, 2000):
        th = rows(np.arange(2 * M + s, 2 * M + min(M, s + 2000))).to(DEV)
        P, obj = sz.weight_match_restarts(th, theta_ref.to(DEV), restarts=args.restarts, seed=s)
        PERMS[s:s + len(th)] = P.cpu(); OBJ[s:s + len(th)] = obj.cpu()
        OBJ0[s:s + len(th)] = sz.alignment_objective(th, theta_ref.to(DEV)).cpu()
        log(f"aligned {s+len(th)}/{M}")
    np.save(perm_file, PERMS.numpy()); np.save("runs/sane_perms_obj.npy", np.stack([OBJ0.numpy(), OBJ.numpy()]))
    log(f"alignment done in {time.time()-t0:.0f}s; objective {OBJ0.mean():.1f} -> {OBJ.mean():.1f}")
perm_of = lambda idx: PERMS[np.asarray(idx) % M]
if args.perms_only:
    log("perms_only: done"); raise SystemExit(0)


def aligned(idx):
    th = rows(idx).to(DEV)
    return sz.permute_batched(th, perm_of(idx).to(DEV))


# ---------------- standardisation statistics (aligned train rows, all checkpoints)
sidx = np.sort(rng.choice(tr, min(6000, len(tr)), replace=False))
mu, sigma = sz.layer_stats(aligned(sidx).cpu())
log("mu", mu.numpy().round(4), "sigma", sigma.numpy().round(4))

# ---------------- model
torch.manual_seed(args.seed)
cfg = sz.ae_config(d_model=args.d_model, nhead=args.nhead, num_layers=args.layers, lat_dim=args.lat)
sane = sz.SaneZoo(cfg, mu, sigma).to(DEV)
n_par = sum(p.numel() for p in sane.parameters())
crit = GammaContrastReconLoss(gamma=args.gamma, reduction="mean", batch_size=args.batch, temperature=args.temperature).to(DEV)
opt = torch.optim.AdamW(sane.parameters(), lr=args.lr, weight_decay=args.wd)
steps_per_epoch = len(tr) // args.batch
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, steps_per_epoch * args.epochs))
mask3 = sane.mask[None].expand(args.batch, -1, -1).float()
use_amp = DEV.type == "cuda"
log(f"params {n_par}; train seq {len(tr)} val {len(va)} test {len(te)}; steps/epoch {steps_per_epoch}")

Xtr_al = aligned(tr)  # keep aligned train rows on device (45k x 4970 fp32 = 0.9 GB)
Xva_al = aligned(va)
gen = torch.Generator(device="cpu").manual_seed(args.seed)


def random_perms(B):
    return torch.stack([torch.stack([torch.randperm(16, generator=gen) for _ in range(3)]) for _ in range(B)]).to(DEV)


def view(theta_al, permute):
    th = sz.permute_batched(theta_al, random_perms(theta_al.shape[0])) if permute else theta_al
    tok = sane.tokens(th)
    if args.noise > 0:
        tok = tok * (1.0 + args.noise * torch.randn_like(tok)) * sane.mask
    return tok


@torch.no_grad()
def eval_recon(X_al, n=2048):
    sane.eval(); tot, rsq, cnt = 0.0, 0.0, 0
    for s in range(0, min(n, len(X_al)), 256):
        tok = sane.tokens(X_al[s:s + 256])
        z, y, _ = sane(tok)
        m = sane.mask[None].expand_as(tok).float()
        out = crit.loss_recon(y.float(), tok, m)
        tot += out["loss_recon"].item() * len(tok); rsq += float(out["rsq"]) * len(tok); cnt += len(tok)
    sane.train(); return tot / cnt, rsq / cnt


hist = []
t_train = time.time()
for ep in range(args.epochs):
    sane.train(); t_ep = time.time(); run = {"loss": 0.0, "recon": 0.0, "contrast": 0.0}; nb = 0
    order = torch.randperm(len(tr), generator=gen)
    for b in range(steps_per_epoch):
        idx = order[b * args.batch:(b + 1) * args.batch].to(DEV)
        th = Xtr_al[idx]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            x_i, x_j = view(th, permute=True), view(th, permute=False)   # view 1 permuted, view 2 canonical
            z_i, y_i, zp_i = sane(x_i); z_j, y_j, zp_j = sane(x_j)
            perf = crit(z_i=zp_i.float(), z_j=zp_j.float(), y=torch.cat([y_i, y_j]).float(),
                        t=torch.cat([x_i, x_j]).float(), m=torch.cat([mask3, mask3]))
            loss = perf["loss/loss"]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(sane.parameters(), 1.0); opt.step(); sched.step()
        run["loss"] += loss.item(); run["recon"] += perf["loss/loss_recon"].item(); run["contrast"] += float(perf["loss/loss_contrast"]); nb += 1
    vl, vr = eval_recon(Xva_al)
    h = {k: v / max(nb, 1) for k, v in run.items()}; h.update(epoch=ep + 1, val_recon=vl, val_rsq=vr, epoch_time=time.time() - t_ep, time=time.time() - t_train)
    hist.append(h); log(f"ep {ep+1} loss {h['loss']:.4f} recon {h['recon']:.4f} contrast {h['contrast']:.3f} | val recon {vl:.4f} rsq {vr:.3f} | {h['epoch_time']:.0f}s")
    if args.time_limit and time.time() - T0 > args.time_limit:
        log("time limit reached"); break
train_time = time.time() - t_train

# ---------------- embeddings (clean canonical view) for heads
@torch.no_grad()
def embed(X_al, chunk=512):
    sane.eval(); out = []
    for s in range(0, len(X_al), chunk):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            z = sane.encode(X_al[s:s + chunk])
        out.append(z.float())
    return torch.cat(out)


Ztr, Zva = embed(Xtr_al), embed(Xva_al)
Xte_al = aligned(te); Zte = embed(Xte_al)
del Xtr_al, Xva_al, Xte_al; opt.zero_grad(set_to_none=True); del opt
if DEV.type == "cuda":
    torch.cuda.empty_cache()
Ytr, Yva, Yte = [torch.from_numpy(recall_all[i]).to(DEV) for i in (tr, va, te)]
Atr, Ava, Ate = [torch.from_numpy(act_all[i]).to(DEV) for i in (tr, va, te)]
log(f"embeddings train {tuple(Ztr.shape)} test {tuple(Zte.shape)}")


def fit_linear(head):
    """Closed-form ridge on [z̄, act]; alpha chosen on val."""
    f = lambda Z, A: torch.cat([Z.mean(1), sz.act_onehot(A)], 1)
    Ftr, Fva = f(Ztr, Atr).double(), f(Zva, Ava).double()
    m, s = Ftr.mean(0), Ftr.std(0) + 1e-8
    Fs, Fvs = (Ftr - m) / s, (Fva - m) / s
    ym = Ytr.double().mean(0)
    best = None
    for alpha in [1e-3, 1e-2, 1e-1, 1, 10, 100]:
        W = torch.linalg.solve(Fs.T @ Fs + alpha * torch.eye(Fs.shape[1], device=DEV, dtype=torch.float64), Fs.T @ (Ytr.double() - ym))
        pv = Fvs @ W + ym
        r2v = mc.r2(Yva.cpu().numpy(), pv.float().cpu().numpy())
        if best is None or r2v > best[0]:
            best = (r2v, alpha, W.clone())
    r2v, alpha, W = best
    Wf = (W / s[:, None]).float(); bf = (ym - (m / s) @ W).float()
    with torch.no_grad():
        head.net.weight.copy_(Wf.T); head.net.bias.copy_(bf)
    return {"alpha": alpha, "best_val_r2": r2v}


def fit_head(head, epochs):
    opt_h = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5)
    best, best_sd = -1e9, None
    for ep in range(epochs):
        head.train(); perm = torch.randperm(len(Ztr), device=DEV)
        for s in range(0, len(Ztr), 256):
            i = perm[s:s + 256]
            loss = ((head(Ztr[i], Atr[i]) - Ytr[i]) ** 2).mean()
            opt_h.zero_grad(); loss.backward(); opt_h.step()
        head.eval()
        with torch.no_grad():
            r2v = mc.r2(Yva.cpu().numpy(), head(Zva, Ava).cpu().numpy())
        if r2v > best:
            best, best_sd = r2v, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_sd); head.eval()
    return {"best_val_r2": best}


heads, head_meta = {}, {}
torch.manual_seed(args.seed + 1)
heads["linear"] = sz.ZBarHead(args.lat, kind="linear").to(DEV); head_meta["linear"] = fit_linear(heads["linear"])
heads["mlp"] = sz.ZBarHead(args.lat, kind="mlp").to(DEV); head_meta["mlp"] = fit_head(heads["mlp"], args.head_epochs)
heads["token"] = sz.TokenHead(args.lat).to(DEV); head_meta["token"] = fit_head(heads["token"], args.head_epochs)
for k, h in heads.items():
    with torch.no_grad():
        Pte = h(Zte, Ate).cpu().numpy()
    res = {"spec": f"sane_{k}", "seed": args.seed, "epochs": args.epochs, "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "params": n_par + sum(p.numel() for p in h.parameters()), "best_val_r2": head_meta[k]["best_val_r2"], "train_time_s": train_time,
           "steps_trained": [0, 20, 86], "sane_checkpoint": f"runs/{name}.pt", "head": k, "aligned": not args.no_align}
    res.update(mc.metric_table(recall_all[te], Pte, steps_all[te], acc_all[te], act_all[te], prefix="test_"))
    hname = f"sane_{k}{tag}_s{args.seed}_e{args.epochs}" + (f"_small{args.small}" if args.small else "")
    json.dump(res, open(f"runs/{hname}.json", "w"), indent=1); np.save(f"runs/{hname}_pred_test.npy", Pte.astype(np.float32))
    log(f"head {k}: test r2_all {res['test_r2_all']:.4f} r2_final {res['test_r2_final']:.4f} r2_centred {res['test_r2_centred_final_trained']:.4f}")

# ---------------- identity control: decode(encode(theta)) vs theta on real test images (epoch-86 test models)
Xtr_u8, ytr_img, Xte_u8, yte_img = cnnzoo.load_images(ZOO)
g = np.random.default_rng(12345); per = 200
sub = np.concatenate([g.choice(np.where(yte_img == k)[0], per, replace=False) for k in range(10)])
Ximg = cnnzoo.prep_images(Xte_u8[sub]).to(DEV); yimg = torch.as_tensor(yte_img[sub].astype(np.int64), device=DEV)
te86 = te[steps_all[te] == 86]
te86 = te86[acc_all[te86] >= 0.5] if (acc_all[te86] >= 0.5).sum() >= args.identity_n else te86
pick = np.sort(np.random.default_rng(7).choice(te86, min(args.identity_n, len(te86)), replace=False))
th0 = rows(pick).to(DEV); P0 = perm_of(pick).to(DEV); a0 = torch.from_numpy(act_all[pick]).to(DEV)
with torch.no_grad():
    th_al = sz.permute_batched(th0, P0)
    rec_al = sane.reconstruct(th_al)
    th_rec = sz.permute_batched(rec_al, sz.invert_perms(P0))
    assert torch.allclose(sz.permute_batched(th_al, sz.invert_perms(P0)), th0, atol=1e-6)
    r0, acc0 = cnnzoo.per_class_recall(th0, a0, Ximg, yimg); r1, acc1 = cnnzoo.per_class_recall(th_rec, a0, Ximg, yimg)
    rel_err = ((rec_al - th_al).norm(dim=1) / th_al.norm(dim=1))
d = (r1 - r0).abs()
identity = {"n": int(len(pick)), "rel_recon_err_median": rel_err.median().item(), "rel_recon_err_mean": rel_err.mean().item(),
            "abs_drecall_median": d.median().item(), "abs_drecall_mean": d.mean().item(), "abs_drecall_p90": d.flatten().quantile(0.9).item(),
            "dacc_mean": (acc1 - acc0).mean().item(), "acc_before_mean": acc0.mean().item(), "acc_after_mean": acc1.mean().item(),
            "frac_models_max_drecall_gt_0.05": (d.max(1).values > 0.05).float().mean().item(),
            "gate_passed_median_le_0.05": bool(d.median().item() <= 0.05)}
log("identity control", json.dumps(identity))

sz.save_sane(f"runs/{name}.pt", sane, heads, theta_ref, ref_idx,
             meta={"args": vars(args), "cfg": cfg, "perm_file": perm_file, "head_meta": head_meta, "zoo": ZOO})
out = {"name": name, "zoo": ZOO, "seed": args.seed, "epochs_done": len(hist), "params_ae": n_par, "cfg": cfg, "args": vars(args),
       "aligned": not args.no_align, "ref_idx": int(ref_idx), "mu": mu.tolist(), "sigma": sigma.tolist(),
       "train_time_s": train_time, "epoch_time_s": float(np.mean([h["epoch_time"] for h in hist])) if hist else None,
       "final": hist[-1] if hist else None, "hist": hist, "heads": head_meta, "identity_control": identity,
       "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)), "total_time_s": time.time() - T0}
if os.path.exists("runs/sane_perms_obj.npy"):
    o = np.load("runs/sane_perms_obj.npy"); out["alignment_objective_identity_mean"] = float(o[0].mean()); out["alignment_objective_aligned_mean"] = float(o[1].mean())
json.dump(out, open(f"runs/{name}.json", "w"), indent=1)
log(f"DONE val recon {hist[-1]['val_recon'] if hist else None} identity median |Δrecall| {identity['abs_drecall_median']:.4f} total {time.time()-T0:.0f}s")

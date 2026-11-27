"""SANE (vendored MIT AE, Schürholt et al. ICML 2024) on a 2-D toy zoo (2-16-16-6 ReLU MLP, 422 params).

Tokenisation: one token per output unit, [incoming weights, bias], zero-padded to D_T = 17:
  layer 0: 16 tokens of 2+1, layer 1: 16 tokens of 16+1, layer 2 (output): 6 tokens of 16+1  -> 38 tokens.
Alignment: git-re-basin weight matching of the two hidden-unit permutations to one reference model (final
checkpoint, best train-split accuracy); exact symmetry for ReLU. Augmentation: random permutation view + x(1+0.1 eps).
Heads on z: ridge on z̄, MLP on z̄, per-class token head [z̄, z_out_c, onehot c]. Baseline: ridge on aligned θ.
Identity control: recall of decode(encode(θ)) vs θ on toyzoo.make_test_set() (matches stored recall exactly).

usage: TOYTASK=rings python sane_toy_pretrain.py --zoo zoo.npz --train_min 7 [--d_model 128 --layers 3 --lat 16 ...]
Writes ./out/{summary.json, hist.json, preds_test.npz, sane.pt}.
"""
import os, sys, json, time, math, argparse
os.environ.setdefault("KMP_AFFINITY", "disabled")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import toyzoo as tz
from sane_vendor.def_AE import AE
from sane_vendor.def_loss import GammaContrastReconLoss

ap = argparse.ArgumentParser()
ap.add_argument("--zoo", required=True); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--train_min", type=float, default=7.0); ap.add_argument("--max_epochs", type=int, default=1000)
ap.add_argument("--batch", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=3e-9)
ap.add_argument("--d_model", type=int, default=128); ap.add_argument("--layers", type=int, default=3)
ap.add_argument("--nhead", type=int, default=4); ap.add_argument("--lat", type=int, default=16)
ap.add_argument("--gamma", type=float, default=0.05); ap.add_argument("--temperature", type=float, default=0.1)
ap.add_argument("--noise", type=float, default=0.1); ap.add_argument("--restarts", type=int, default=4)
ap.add_argument("--small", type=int, default=0); ap.add_argument("--head_epochs", type=int, default=40)
ap.add_argument("--identity_n", type=int, default=500); ap.add_argument("--out", default="out")
args = ap.parse_args()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.out, exist_ok=True)
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

H, K = tz.H, tz.K
D_T, N_TOK = H + 1, 2 * H + K
LAYER_FAN = [(tz.D_IN, H), (H, H), (H, K)]
OUT_TOK = slice(2 * H, N_TOK)
MASK = torch.zeros(N_TOK, D_T, dtype=torch.bool); POS = torch.zeros(N_TOK, 3, dtype=torch.long)
LID = torch.zeros(N_TOK, dtype=torch.long)
n = 0
for l, (i, o) in enumerate(LAYER_FAN):
    for r in range(o):
        MASK[n, : i + 1] = True; POS[n] = torch.tensor([n, l, r]); LID[n] = l; n += 1


def tokenize(theta):
    ps = tz.unflatten(theta); out = []
    for l, (i, o) in enumerate(LAYER_FAN):
        t = torch.cat([ps[2 * l].transpose(1, 2), ps[2 * l + 1][:, :, None]], 2)  # (B,o,i+1)
        out.append(F.pad(t, (0, D_T - i - 1)))
    return torch.cat(out, 1)


def untokenize(tok):
    ps, n = [], 0
    for l, (i, o) in enumerate(LAYER_FAN):
        t = tok[:, n:n + o]; n += o
        ps.append(t[:, :, :i].transpose(1, 2).contiguous()); ps.append(t[:, :, i].contiguous())
    return tz.flatten(ps)


def permute(theta, P):
    """P (B,2,H): aligned unit j = original unit P[:,l,j]."""
    W0, b0, W1, b1, W2, b2 = tz.unflatten(theta); B = theta.shape[0]
    p0, p1 = P[:, 0], P[:, 1]
    W0 = W0.gather(2, p0[:, None, :].expand(B, W0.shape[1], H)); b0 = b0.gather(1, p0)
    W1 = W1.gather(1, p0[:, :, None].expand(B, H, H)).gather(2, p1[:, None, :].expand(B, H, H)); b1 = b1.gather(1, p1)
    W2 = W2.gather(1, p1[:, :, None].expand(B, H, K))
    return tz.flatten([W0, b0, W1, b1, W2, b2])


def invert(P):
    inv = torch.empty_like(P); inv.scatter_(2, P, torch.arange(H, device=P.device).expand_as(P)); return inv


def ident(B, dev="cpu"):
    return torch.arange(H, device=dev).view(1, 1, H).expand(B, 2, H).clone()


def objective(theta, ref):
    return (theta * ref[None]).sum(1)  # all params permutation-sensitive except b2 (constant)


def weight_match(theta, ref, max_iter=30, seed=0):
    B = theta.shape[0]; W0, b0, W1, b1, W2, _ = tz.unflatten(theta)
    R0, rb0, R1, rb1, R2, _ = [x[0] for x in tz.unflatten(ref[None])]
    P = ident(B); g = np.random.default_rng(seed); active = np.ones(B, bool)
    in0 = torch.cat([W0, b0[:, None]], 1)                      # (B,3,H) columns = units
    rin0 = torch.cat([R0, rb0[None]], 0)                        # (3,H)
    for it in range(max_iter):
        changed = np.zeros(B, bool)
        for l in g.permutation(2):
            if l == 0:
                W1p = W1.gather(2, P[:, 1][:, None, :].expand(B, H, H))       # cols aligned by p1
                C = torch.einsum("bki,kj->bji", in0, rin0) + torch.einsum("bio,jo->bji", W1p, R1)
            else:
                W1r = W1.gather(1, P[:, 0][:, :, None].expand(B, H, H))       # rows aligned by p0
                in1 = torch.cat([W1r, b1[:, None]], 1); rin1 = torch.cat([R1, rb1[None]], 0)
                C = torch.einsum("bki,kj->bji", in1, rin1) + torch.einsum("bio,jo->bji", W2, R2)
            Cn = C.numpy()
            for b in np.flatnonzero(active):
                _, col = linear_sum_assignment(Cn[b], maximize=True)
                col = torch.from_numpy(col)
                if not torch.equal(col, P[b, l]):
                    P[b, l] = col; changed[b] = True
        active &= changed
        if not active.any():
            break
    return P


def weight_match_restarts(theta, ref, restarts, seed):
    B = theta.shape[0]; gen = torch.Generator().manual_seed(seed); best, bobj = None, None
    for r in range(restarts):
        P0 = ident(B) if r == 0 else torch.stack([torch.stack([torch.randperm(H, generator=gen) for _ in range(2)]) for _ in range(B)])
        P = weight_match(permute(theta, P0), ref, seed=seed + r)
        Pc = P0.gather(2, P); obj = objective(permute(theta, Pc), ref)
        if best is None:
            best, bobj = Pc, obj
        else:
            m = obj > bobj; best[m] = Pc[m]; bobj = torch.where(m, obj, bobj)
    return best, bobj


# ---------------------------------------------------------------- data
Z = np.load(args.zoo)
TH = Z["theta"].astype(np.float32); REC = Z["recall"].astype(np.float32); ACC = Z["acc"].astype(np.float32)
NC, M, _ = TH.shape; split = Z["split"]
TASKNAME = str(Z["task"]) if "task" in Z.files else os.environ.get("TOYTASK", "blobs")
log(f"zoo {args.zoo} task {TASKNAME} ckpts {Z['ckpt_steps'].tolist()} models {M} device {DEV}")
models = {s: np.where(split == s)[0] for s in ("train", "val", "test")}
if args.small:
    models = {s: np.sort(rng.choice(v, min(len(v), args.small if s == "train" else args.small // 2), replace=False)) for s, v in models.items()}
ref_idx = int(models["train"][np.argmax(ACC[-1, models["train"]])])
ref = torch.from_numpy(TH[-1, ref_idx])

# alignment on the final checkpoint, reused for earlier checkpoints (as in the Small-CNN pipeline)
use = np.sort(np.concatenate(list(models.values())))
th_final = torch.from_numpy(TH[-1, use])
t0 = time.time(); PERM = torch.zeros(M, 2, H, dtype=torch.long); obj0 = objective(th_final, ref); objA = torch.zeros(len(use))
for s in range(0, len(use), 5000):
    P, o = weight_match_restarts(th_final[s:s + 5000], ref, args.restarts, seed=s)
    PERM[use[s:s + 5000]] = P; objA[s:s + 5000] = o
log(f"alignment {len(use)} models in {time.time()-t0:.0f}s; objective {obj0.mean():.2f} -> {objA.mean():.2f}")
# check function invariance of the permutation on a few models
Xte, yte = tz.make_test_set()
chk = use[:200]
r_a, _ = tz.per_class_recall(permute(torch.from_numpy(TH[-1, chk]), PERM[chk]), Xte, yte)
assert np.abs(r_a.numpy() - REC[-1, chk]).max() == 0, "permutation changed the function"


def rows(split_name):
    idx = models[split_name]
    th = torch.from_numpy(TH[:, idx].reshape(-1, TH.shape[2]))
    P = PERM[idx].repeat(NC, 1, 1)
    al = permute(th, P)
    return al, REC[:, idx].reshape(-1, K), ACC[:, idx].reshape(-1), np.repeat(np.arange(NC), len(idx)), np.tile(idx, NC)


Xtr, Ytr, Atr, Ctr, _ = rows("train"); Xva, Yva, Ava, Cva, _ = rows("val"); Xte_al, Yte, Ate, Cte, Mte = rows("test")
log(f"sequences train {len(Xtr)} val {len(Xva)} test {len(Xte_al)}")

# standardisation (per-layer scalar mu/sigma over weights+bias, repo style)
tok = tokenize(Xtr[rng.choice(len(Xtr), min(20000, len(Xtr)), replace=False)])
mu = torch.zeros(3); sg = torch.zeros(3)
for l, (i, o) in enumerate(LAYER_FAN):
    rws = tok[:, LID == l, : i + 1].reshape(tok.shape[0], -1)
    mu[l] = rws.mean(1).mean(); sg[l] = rws.std(1).pow(2).mean().sqrt()
log("mu", mu.numpy().round(4), "sigma", sg.numpy().round(4))
mu_t, sg_t = mu[LID].view(1, N_TOK, 1).to(DEV), sg[LID].view(1, N_TOK, 1).to(DEV)
MASKd, POSd = MASK.to(DEV), POS.to(DEV)
std = lambda t: (t - mu_t) / sg_t * MASKd
destd = lambda t: (t * sg_t + mu_t) * MASKd

# ---------------------------------------------------------------- model
cfg = {"ae:i_dim": D_T, "ae:d_model": args.d_model, "ae:nhead": args.nhead, "ae:num_layers": args.layers,
       "ae:lat_dim": args.lat, "training::windowsize": N_TOK, "ae:dropout": 0.0,
       "ae:max_positions": [N_TOK, 3, H], "ae:transformer_type": "gpt2"}
ae = AE(cfg).to(DEV)
n_par = sum(p.numel() for p in ae.parameters())
log(f"SANE AE params {n_par:,} (d_model {args.d_model}, {args.layers}+{args.layers} layers, {args.nhead} heads, lat {args.lat}, {N_TOK} tokens x {D_T})")
crit = GammaContrastReconLoss(gamma=args.gamma, reduction="mean", batch_size=args.batch, temperature=args.temperature).to(DEV)
opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, weight_decay=args.wd)
pos = lambda B: POSd[None].expand(B, N_TOK, 3)
Xtr_d, Xva_d = Xtr.to(DEV), Xva.to(DEV)
gen = torch.Generator().manual_seed(args.seed)


def rand_perm(B):
    return torch.argsort(torch.rand(B, 2, H, device=DEV), dim=2)


def view(th, perm):
    t = std(tokenize(permute(th, rand_perm(th.shape[0])) if perm else th))
    return t * (1 + args.noise * torch.randn_like(t)) * MASKd if args.noise > 0 else t


@torch.no_grad()
def eval_recon(X, n=4096):
    ae.eval(); tot = rsq = cnt = 0.0
    for s in range(0, min(n, len(X)), 512):
        t = std(tokenize(X[s:s + 512])); z, y, _ = ae(t, pos(len(t)))
        o = crit.loss_recon(y.float(), t, MASKd[None].expand_as(t).float())
        tot += o["loss_recon"].item() * len(t); rsq += float(o["rsq"]) * len(t); cnt += len(t)
    ae.train(); return tot / cnt, rsq / cnt


budget = args.train_min * 60; spe = len(Xtr) // args.batch
mask3 = MASKd[None].expand(args.batch, -1, -1).float(); use_amp = DEV.type == "cuda"
hist, step, t_tr = [], 0, time.time(); stop = False
for ep in range(args.max_epochs):
    ae.train(); te0 = time.time(); run = np.zeros(3); nb = 0
    order = torch.randperm(len(Xtr), generator=gen).to(DEV)
    for b in range(spe):
        frac = (time.time() - t_tr) / budget
        if frac >= 1:
            stop = True; break
        lr = args.lr * (frac / 0.05 if frac < 0.05 else 0.5 * (1 + math.cos(math.pi * (frac - 0.05) / 0.95)))
        for gp in opt.param_groups:
            gp["lr"] = lr
        th = Xtr_d[order[b * args.batch:(b + 1) * args.batch]]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            xi, xj = view(th, True), view(th, False)
            zi, yi, zpi = ae(xi, pos(len(xi))); zj, yj, zpj = ae(xj, pos(len(xj)))
            perf = crit(z_i=zpi.float(), z_j=zpj.float(), y=torch.cat([yi, yj]).float(), t=torch.cat([xi, xj]).float(), m=torch.cat([mask3, mask3]))
            loss = perf["loss/loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0); opt.step()
        run += [loss.item(), perf["loss/loss_recon"].item(), float(perf["loss/loss_contrast"])]; nb += 1; step += 1
    vl, vr = eval_recon(Xva_d)
    h = dict(epoch=ep + 1, steps=step, loss=run[0] / max(nb, 1), recon=run[1] / max(nb, 1), contrast=run[2] / max(nb, 1),
             val_recon=vl, val_rsq=vr, epoch_time=time.time() - te0, partial=stop)
    hist.append(h); log(f"ep {ep+1} ({nb} steps) loss {h['loss']:.4f} recon {h['recon']:.4f} contr {h['contrast']:.3f} | val recon {vl:.4f} rsq {vr:.3f} | {h['epoch_time']:.0f}s")
    if stop:
        break
train_time = time.time() - t_tr


# ---------------------------------------------------------------- embeddings + heads
@torch.no_grad()
def embed(X):
    ae.eval(); out = []
    for s in range(0, len(X), 4096):
        t = std(tokenize(X[s:s + 4096].to(DEV)))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            out.append(ae.forward_encoder(t, pos(len(t))).float())
    return torch.cat(out)


Ztr, Zva, Zte = embed(Xtr), embed(Xva), embed(Xte_al)
Yt = [torch.from_numpy(y).to(DEV) for y in (Ytr, Yva, Yte)]


def r2(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64); return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def dc(y):
    return y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()


def metrics(P):
    f = (Cte == NC - 1); t = f & (Ate >= 0.5)
    Y = Yte
    return {"r2_all": r2(Y, P), "mae_all": float(np.abs(Y - P).mean()), "r2_final": r2(Y[f], P[f]),
            "r2_final_trained": r2(Y[t], P[t]),
            "r2_model_centred": r2(Y[t] - Y[t].mean(1, keepdims=True), P[t] - P[t].mean(1, keepdims=True)),
            "r2_interaction": r2(dc(Y[t]), dc(P[t])), "n_final_trained": int(t.sum())}


def ridge(Ftr, Fva, Fte, name):
    Ftr, Fva, Fte = [torch.as_tensor(x, device=DEV).double() for x in (Ftr, Fva, Fte)]
    m, s = Ftr.mean(0), Ftr.std(0) + 1e-8; A, Bv, Ct = (Ftr - m) / s, (Fva - m) / s, (Fte - m) / s
    ym = Yt[0].double().mean(0); best = None
    for a in [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]:
        W = torch.linalg.solve(A.T @ A + a * torch.eye(A.shape[1], device=DEV, dtype=torch.float64), A.T @ (Yt[0].double() - ym))
        v = r2(Yva, (Bv @ W + ym).cpu().numpy())
        if best is None or v > best[0]:
            best = (v, a, W)
    P = (Ct @ best[2] + ym).float().cpu().numpy(); log(f"{name}: alpha {best[1]} val r2 {best[0]:.4f}")
    return P, {"alpha": best[1], "val_r2": best[0]}


class ZbarMLP(nn.Module):
    def __init__(s, lat, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(lat, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, K))
    def forward(s, z): return torch.sigmoid(s.net(z.mean(1)))


class TokenHead(nn.Module):
    def __init__(s, lat, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * lat + K, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(s, z):
        B = z.shape[0]
        x = torch.cat([z.mean(1, keepdim=True).expand(B, K, z.shape[2]), z[:, OUT_TOK], torch.eye(K, device=z.device)[None].expand(B, K, K)], 2)
        return torch.sigmoid(s.net(x)[..., 0])


def fit(head, epochs):
    head.to(DEV); o = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-5); best = (-1e9, None)
    for ep in range(epochs):
        head.train(); perm = torch.randperm(len(Ztr), device=DEV)
        for s in range(0, len(Ztr), 512):
            i = perm[s:s + 512]; l = ((head(Ztr[i]) - Yt[0][i]) ** 2).mean(); o.zero_grad(); l.backward(); o.step()
        head.eval()
        with torch.no_grad():
            v = r2(Yva, head(Zva).cpu().numpy())
        if v > best[0]:
            best = (v, {k: x.clone() for k, x in head.state_dict().items()})
    head.load_state_dict(best[1]); head.eval()
    with torch.no_grad():
        return head(Zte).cpu().numpy(), {"val_r2": best[0]}


res, preds, HEADS = {}, {}, {"sane_mlp": ZbarMLP(args.lat), "sane_token": TokenHead(args.lat)}
th0 = time.time()
for name, fn in [("sane_linear", lambda: ridge(Ztr.mean(1).cpu(), Zva.mean(1).cpu(), Zte.mean(1).cpu(), "sane_linear")),
                 ("sane_mlp", lambda: fit(HEADS["sane_mlp"], args.head_epochs)),
                 ("sane_token", lambda: fit(HEADS["sane_token"], args.head_epochs)),
                 ("ridge_theta_aligned", lambda: ridge(Xtr, Xva, Xte_al, "ridge_theta_aligned"))]:
    P, meta = fn(); preds[name] = P.astype(np.float32); res[name] = {**meta, **metrics(P)}
    log(name, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name].items()})
res["zoo_mean_const"] = metrics(np.broadcast_to(Ytr.mean(0), Yte.shape).copy())
head_time = time.time() - th0

# ---------------------------------------------------------------- identity control (final ckpt, trained test models)
cand = models["test"][ACC[-1, models["test"]] >= 0.5]
pick = np.sort(np.random.default_rng(7).choice(cand, min(args.identity_n, len(cand)), replace=False))
th_o = torch.from_numpy(TH[-1, pick]); Pp = PERM[pick]; al = permute(th_o, Pp)
with torch.no_grad():
    ae.eval(); t = std(tokenize(al.to(DEV))); z = ae.forward_encoder(t, pos(len(t))); y = ae.forward_decoder(z, pos(len(t)))
    rec = untokenize(destd(y).cpu()); back = permute(rec, invert(Pp))
assert torch.allclose(untokenize(tokenize(al)), al) and torch.allclose(permute(al, invert(Pp)), th_o, atol=1e-6)
r0, a0 = tz.per_class_recall(th_o, Xte, yte); r1, a1 = tz.per_class_recall(back, Xte, yte)
d = (r1 - r0).abs(); rel = (rec - al).norm(dim=1) / al.norm(dim=1)
identity = {"n": int(len(pick)), "rel_weight_err_median": rel.median().item(), "abs_drecall_median": d.median().item(),
            "abs_drecall_mean": d.mean().item(), "abs_drecall_p90": d.flatten().quantile(0.9).item(),
            "acc_before": a0.mean().item(), "acc_after": a1.mean().item(), "frac_models_max_drecall_gt_0.05": (d.max(1).values > 0.05).float().mean().item()}
log("identity", identity)

summary = {"zoo": os.path.basename(args.zoo), "task": TASKNAME, "args": vars(args), "cfg": cfg, "params_ae": n_par,
           "n_tokens": N_TOK, "token_width": D_T, "ref_idx": ref_idx, "align_obj_identity": obj0.mean().item(), "align_obj_aligned": objA.mean().item(),
           "steps": step, "epochs": len(hist), "train_time_s": train_time, "head_time_s": head_time, "final": hist[-1] if hist else None,
           "results": res, "identity": identity, "total_time_s": time.time() - T0, "device": str(DEV),
           "gpu": torch.cuda.get_device_name() if DEV.type == "cuda" else None}
json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1); json.dump(hist, open(f"{args.out}/hist.json", "w"), indent=1)
np.savez_compressed(f"{args.out}/preds_test.npz", Y=Yte, ckpt=Cte, acc=Ate, model=Mte, **preds)
torch.save({"cfg": cfg, "ae": ae.state_dict(), "mu": mu, "sigma": sg, "ref_idx": ref_idx, "perm": PERM,
            "heads": {k: v.state_dict() for k, v in HEADS.items()}, "args": vars(args)}, f"{args.out}/sane.pt")
log(f"DONE total {time.time()-T0:.0f}s")

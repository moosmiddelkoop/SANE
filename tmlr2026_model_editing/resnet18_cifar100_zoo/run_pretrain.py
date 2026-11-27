"""Windowed SANE pretraining on the ResNet-18 zoo. usage:
python run_pretrain.py ZOO_ROOT SEED EPOCHS [--d_model 1024 --layers 4 --lat 128 --window 512 --batch 64 --lr 1e-4 --gamma 0.05
   --noise 0.1 --n_train 0 --steps_per_epoch 0 --time_limit S --tag T --no_embed]
Writes runs/sane<tag>_s<seed>.pt, runs/sane<tag>_s<seed>.json (history), runs/sane<tag>_s<seed>_emb.npz (z̄, per-layer means, fc-token latents for all models)."""
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.append(os.getcwd())
import numpy as np, torch
import resnet_zoo as rz, sane_resnet as sr
from sane_vendor.def_loss import GammaContrastReconLoss
ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("seed", type=int); ap.add_argument("epochs", type=int)
ap.add_argument("--d_model", type=int, default=1024); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--lat", type=int, default=128)
ap.add_argument("--nhead", type=int, default=8); ap.add_argument("--window", type=int, default=512); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=3e-9); ap.add_argument("--gamma", type=float, default=0.05)
ap.add_argument("--temperature", type=float, default=0.1); ap.add_argument("--noise", type=float, default=0.1)
ap.add_argument("--n_train", type=int, default=0); ap.add_argument("--steps_per_epoch", type=int, default=0); ap.add_argument("--time_limit", type=float, default=None)
ap.add_argument("--tag", default=""); ap.add_argument("--no_embed", action="store_true"); ap.add_argument("--n_val", type=int, default=20)
args = ap.parse_args()
dev = rz.DEV; name = f"sane{args.tag}_s{args.seed}"; os.makedirs("runs", exist_ok=True); T0 = time.time()
log = lambda *x: print(f"[{name} {time.time()-T0:6.0f}s]", *x, flush=True)
zoo = rz.Zoo(args.root); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
tr = zoo.idx("train"); va = zoo.idx("val")
if args.n_train: tr = np.sort(rng.choice(tr, args.n_train, replace=False))
log(f"loading {len(tr)} train models' tokens to {dev} ...")
t0 = time.time(); TOK = torch.empty(len(tr), rz.N_TOK, rz.D_T, dtype=torch.float16, device=dev)
for i, m in enumerate(tr): TOK[i] = torch.from_numpy(np.array(zoo.tokens[m])).to(dev)
VAL = torch.from_numpy(np.array(zoo.tokens[va[: args.n_val]])).to(dev)
log(f"tokens loaded in {time.time()-t0:.0f}s: {tuple(TOK.shape)} ({TOK.numel()*2/1e9:.1f} GB)")
cfg = sr.ae_config(args.d_model, args.nhead, args.layers, args.lat, args.window)
sane = sr.SaneResNet(cfg).to(dev); n_par = sum(p.numel() for p in sane.parameters())
crit = GammaContrastReconLoss(gamma=args.gamma, reduction="mean", batch_size=args.batch, temperature=args.temperature).to(dev)
opt = torch.optim.AdamW(sane.parameters(), lr=args.lr, weight_decay=args.wd)
W = args.window; n_win = int(np.ceil(rz.N_TOK / W))
spe = args.steps_per_epoch or (len(tr) * n_win) // args.batch
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, spe * args.epochs))
MASK = sane.mask; POS = sane.pos
log(f"AE params {n_par/1e6:.1f}M; steps/epoch {spe}; windows/model {n_win}")


def sample_batch(B):
    m = torch.randint(0, len(tr), (B,), device=dev); s = torch.randint(0, rz.N_TOK - W + 1, (B,), device=dev)
    ar = torch.arange(W, device=dev)
    idx = s[:, None] + ar[None]                      # (B, W) token indices
    x = TOK[m[:, None], idx].float()                 # (B, W, D_T)
    return x, POS[idx], MASK[idx].float()


@torch.no_grad()
def eval_val():
    sane.eval(); tot = rsq = cnt = 0
    for b in range(VAL.shape[0]):
        for s in range(0, rz.N_TOK - W + 1, W):
            x = VAL[b, s:s + W][None].float(); p = POS[s:s + W][None]; mk = MASK[s:s + W][None].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z, y, _ = sane(x, p)
            o = crit.loss_recon(y.float(), x, mk); tot += o["loss_recon"].item(); rsq += float(o["rsq"]); cnt += 1
    sane.train(); return tot / cnt, rsq / cnt


hist = []; t_train = time.time()
for ep in range(args.epochs):
    run = {"loss": 0.0, "recon": 0.0, "contrast": 0.0}; t_ep = time.time()
    for it in range(spe):
        x, p, mk = sample_batch(args.batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x_i = x * (1.0 + args.noise * torch.randn_like(x)) * mk if args.noise > 0 else x
            z_i, y_i, zp_i = sane(x_i, p); z_j, y_j, zp_j = sane(x, p)
            perf = crit(z_i=zp_i.float(), z_j=zp_j.float(), y=torch.cat([y_i, y_j]).float(), t=torch.cat([x_i, x]).float(), m=torch.cat([mk, mk]))
            loss = perf["loss/loss"]
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(sane.parameters(), 1.0); opt.step(); sched.step()
        run["loss"] += loss.item(); run["recon"] += perf["loss/loss_recon"].item(); run["contrast"] += float(perf["loss/loss_contrast"])
        if ep == 0 and it in (20, 100):
            log(f"it {it}: {(time.time()-t_ep)/(it+1):.3f} s/step; mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB; loss {loss.item():.4f}")
    vl, vr = eval_val(); h = {k: v / spe for k, v in run.items()}
    h.update(epoch=ep + 1, val_recon=vl, val_rsq=vr, epoch_time=time.time() - t_ep, time=time.time() - t_train); hist.append(h)
    log(f"ep {ep+1} loss {h['loss']:.4f} recon {h['recon']:.4f} contrast {h['contrast']:.3f} | val recon {vl:.4f} rsq {vr:.3f} | {h['epoch_time']:.0f}s")
    json.dump({"name": name, "cfg": cfg, "args": vars(args), "params_ae": n_par, "hist": hist, "n_train": int(len(tr))}, open(f"runs/{name}.json", "w"), indent=1)
    if (ep + 1) % 5 == 0 or ep + 1 == args.epochs: sr.save_sane(f"runs/{name}.pt", sane, {}, {"args": vars(args), "cfg": cfg, "epochs_done": ep + 1})
    if args.time_limit and time.time() - T0 > args.time_limit: log("time limit reached"); break
sr.save_sane(f"runs/{name}.pt", sane, {}, {"args": vars(args), "cfg": cfg, "epochs_done": len(hist)})
del TOK, VAL; torch.cuda.empty_cache()
if not args.no_embed:
    sane.eval(); M = zoo.M; lat = args.lat
    zbar = np.zeros((M, lat), np.float32); zlay = np.zeros((M, rz.N_LAYERS, lat), np.float32); zfc = np.zeros((M, rz.NC, rz.TOK_PER_CLASS, lat), np.float32)
    zstd = np.zeros((M, lat), np.float32); t0 = time.time()
    for m in range(M):
        tok = torch.from_numpy(np.array(zoo.tokens[m])).to(dev)[None]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = sane.encode_model(tok)[0]
        zbar[m] = z.mean(0).cpu().numpy(); zstd[m] = z.std(0).cpu().numpy()
        for l, sl in enumerate(rz.TOKENS_OF_LAYER): zlay[m, l] = z[sl].mean(0).cpu().numpy()
        zfc[m] = sr.zfc_from_z(z[None])[0].cpu().numpy()
        if m % 200 == 0: log(f"embedded {m}/{M} ({(time.time()-t0)/(m+1):.2f} s/model)")
    np.savez(f"runs/{name}_emb.npz", zbar=zbar, zstd=zstd, zlay=zlay, zfc=zfc, split=zoo.split, recall=zoo.recall)
    log("embeddings saved")
log(f"DONE epochs {len(hist)} val recon {hist[-1]['val_recon'] if hist else None} total {time.time()-T0:.0f}s")

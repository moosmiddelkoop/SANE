"""Editing worker. Writes results/<tag>.npz with per-edit records.
usage: python run_edit.py METHOD TASK --set test --eta 1e-3 --max_steps 1000 [--n_per_class 100] [--traj] [--judges run1,run2]
METHOD:
  meta:<run>       Algorithm 1 with metanet runs/<run>.pt (normalised step). Also emits derived baselines
                   random_matched / bias_matched (same norm) and bias_line (bias-only direction, metanet stop).
  meta_raw:<run>   paper recipe: raw gradient step eta (default 0.1), no normalisation.
  meta_out:<run>   metanet gradient restricted to the output layer.
  ga_probes:<run>  gradient ascent/descent on the probe net's own probes pseudo-labelled c, metanet stop.
  ga_real          gradient ascent on real forget-class images (unlearn) / descent on target images (uplearn), 20 Adam steps.
  ft_retain        fine-tune on retain classes only (unlearn) / balanced data incl. target (uplearn), 100 Adam steps.
  meta:<sane_run>@<head>   SANE-W: Algorithm 1 through the SANE encoder + head (linear|mlp|token); alignment fixed at theta0.
  zedit:<sane_run>@<head>  SANE-Z: Algorithm 1 in SANE's token-latent space, decoded through the SANE decoder (--eta_z);
                   also emits sane_identity (decode(encode(theta0))) and zrandom (norm-matched random walk in z) controls.
  zedit_dense:<sane_run>@<head>  SANE-Z restricted to the latents of the 10 dense-row tokens.
  zdec:<sane_run>@<head>   decoder-manifold edit: theta = decode(z), loss/stop on psi(decode(z)) with the gradient through
                   the decoder; psi = SANE round trip (encoder+head) or, with --zpsi <run>, an external metanet (e.g. graph net).
  --sane_monitor <sane_run>  log the SANE reconstruction error (manifold distance) along every logged trajectory."""
import os, sys, json, time, argparse
os.environ["KMP_AFFINITY"] = "disabled"; os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
import cnnzoo, meta_cnn as mc, edit_cnn as ec

ap = argparse.ArgumentParser()
ap.add_argument("method"); ap.add_argument("task", choices=["unlearn", "uplearn"])
ap.add_argument("--set", default="test", choices=["val", "test"])
ap.add_argument("--eta", type=float, default=None)
ap.add_argument("--max_steps", type=int, default=2000)
ap.add_argument("--n_per_class", type=int, default=100)
ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--traj", action="store_true", help="log predicted & true target recall every --log_every steps (subset of models, no derived baselines)")
ap.add_argument("--traj_all", action="store_true", help="log trajectories for ALL models and keep derived baselines")
ap.add_argument("--log_every", type=int, default=20)
ap.add_argument("--traj_sub", type=int, default=2000, help="stratified test subset (images) used for the trajectory true-recall logs; 0 = full test set")
ap.add_argument("--judges", default="")
ap.add_argument("--lr_mult", type=float, default=0.01, help="data baselines: Adam lr = lr_mult * rms(theta)")
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no_derived", action="store_true", help="skip random/bias derived baselines")
ap.add_argument("--save_delta", action="store_true", help="store the edit vector theta1-theta0 per edit")
ap.add_argument("--eta_z", type=float, default=1e-2, help="SANE-Z relative step in latent space")
ap.add_argument("--sane_monitor", default="", help="SANE run (runs/<run>.pt) whose reconstruction error is logged along trajectories")
ap.add_argument("--restarts", type=int, default=8, help="weight-matching restarts for SANE alignment")
ap.add_argument("--zpsi", default="", help="zdec: external metanet run scoring decode(z) (default: SANE round trip)")
args = ap.parse_args()
os.makedirs("results", exist_ok=True)
if os.path.exists("job_overrides.json"):   # runtime scope adjustments without restarting the queue
    ov = json.load(open("job_overrides.json"))
    for pat in ov.get("skip_patterns", []):
        if pat in f"{args.method} {args.task}":
            print(f"skipping {args.method} {args.task} (override {pat})"); raise SystemExit(0)
    for k, v in ov.get("n_per_class_override", {}).items():
        if args.method.startswith(k):
            args.n_per_class = min(args.n_per_class, int(v))
tag = args.tag or f"{args.method.replace(':', '-').replace('@', '-')}_{args.task}_{args.set}_n{args.n_per_class}" + (f"_eta{args.eta:g}" if args.eta else "") + ("_traj" if args.traj else "") \
    + ("" if args.classes == "0,1,2,3,4,5,6,7,8,9" else "_c" + args.classes.replace(",", "")) + (f"_zpsi-{args.zpsi}" if args.zpsi else "")
if os.path.exists(f"results/{tag}.npz") and not os.environ.get("FORCE_RERUN"):
    print(f"[{tag}] exists, skipping"); raise SystemExit(0)
t0 = time.time()

# ---------------- data
meta = np.load("zoo_meta.npz", allow_pickle=False)
TH = np.load("zoo_theta.npy", mmap_mode="r"); M = len(meta["model"])
recall86 = meta["recall"][2]; acc86 = meta["acc"][2]; act_m = meta["act"].astype(np.int64); split = meta["split"]
ZOO = os.environ.get("ZOO", "mnist"); DEV = cnnzoo.DEV
Xtr_u8, ytr, Xte_u8, yte = cnnzoo.load_images(ZOO)
Xte = cnnzoo.prep_images(Xte_u8).to(DEV); yte_t = torch.as_tensor(yte.astype(np.int64), device=DEV)
IMG_HW = tuple(Xte.shape[-2:])
if args.traj_sub:
    _g = np.random.default_rng(12345); _per = args.traj_sub // 10
    _sub = np.concatenate([_g.choice(np.where(yte == k)[0], min(_per, int((yte == k).sum())), replace=False) for k in range(10)])
    Xte_sub, yte_sub = Xte[_sub], yte_t[_sub]
else:
    Xte_sub, yte_sub = Xte, yte_t
sets = np.load("edit_sets.npz")
classes = [int(c) for c in args.classes.split(",")]

kind, run = (args.method.split(":") + [None])[:2]
if run and "@" in run:
    _r, _h = run.split("@"); run_path = f"runs/{_r}.pt@{_h}"
else:
    run_path = f"runs/{run}.pt" if run else None
net = ec.load_metanet(run_path) if run else None
psi = ec.psi_fn(net) if net else None
is_sane = net is not None and hasattr(net, "set_perms")
if is_sane:
    net.restarts = args.restarts
monitor = None
if args.sane_monitor:
    _mon_net = ec.load_metanet(f"runs/{args.sane_monitor}.pt@token")
    _mon_net.restarts = args.restarts
    monitor = _mon_net.recon_error      # recomputes the alignment of theta_t at every logged step
judges = {j: ec.load_metanet(f"runs/{j.split('@')[0]}.pt" + (("@" + j.split('@')[1]) if "@" in j else "")) for j in args.judges.split(",") if j and os.path.exists(f"runs/{j.split('@')[0]}.pt")}
for j in args.judges.split(","):
    if j and j not in judges:
        print(f"[warn] judge runs/{j}.pt missing, skipped", flush=True)
gen = torch.Generator().manual_seed(args.seed)
if args.eta is None:
    args.eta = 0.1 if kind == "meta_raw" else 1e-3

records = {}


def add(name, mi, c, th0, th1, a, extra):
    rec0 = torch.as_tensor(recall86[mi], device=DEV); rec1 = ec.true_recall(th1, a, Xte, yte_t)
    sc = ec.scores(rec0, rec1, c, args.task)
    rec0, rec1 = rec0.cpu(), rec1.cpu(); sc = {k: v.cpu() for k, v in sc.items()}
    th0, th1 = th0.cpu(), th1.cpu(); c = c.cpu(); a = a.cpu()
    extra = {k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in extra.items()}
    d = th1 - th0
    dn = d.norm(dim=1) + 1e-12
    S = ec.SEG; ar_ = torch.arange(len(c))
    dw = d[:, S["dw"][0]:S["dw"][1]].reshape(-1, 10, 16); db = d[:, S["db"][0]:S["db"][1]]
    r = dict(model=mi, c=c.numpy(), act=a.numpy(), rec0=rec0.numpy(), rec1=rec1.numpy(),
             delta_norm=dn.numpy(), theta_norm=th0.norm(dim=1).numpy(),
             out_layer_frac=(d[:, S["dw"][0]:].norm(dim=1) / dn).numpy(),
             # squared-norm shares of the edit per parameter group (sum to 1)
             share_conv1=((d[:, S["c1b"][0]:S["c1w"][1]].norm(dim=1) / dn) ** 2).numpy(),
             share_conv2=((d[:, S["c2b"][0]:S["c2w"][1]].norm(dim=1) / dn) ** 2).numpy(),
             share_conv3=((d[:, S["c3b"][0]:S["c3w"][1]].norm(dim=1) / dn) ** 2).numpy(),
             share_dense_w_c=((dw[ar_, c].norm(dim=1) / dn) ** 2).numpy(),            # output weights of the target class
             share_dense_w_other=((dw.norm(dim=(1, 2)) ** 2 - dw[ar_, c].norm(dim=1) ** 2) / dn ** 2).numpy(),
             share_dense_b_c=((db[ar_, c] / dn) ** 2).numpy(),
             share_dense_b_other=((db.norm(dim=1) ** 2 - db[ar_, c] ** 2) / dn ** 2).numpy(),
             sign_b_c=torch.sign(db[ar_, c]).numpy(),
             # relative change of the model's own parameter groups
             rel_change_dense=(d[:, S["dw"][0]:].norm(dim=1) / (th0[:, S["dw"][0]:].norm(dim=1) + 1e-12)).numpy(),
             rel_change_conv=(d[:, :S["dw"][0]].norm(dim=1) / (th0[:, :S["dw"][0]].norm(dim=1) + 1e-12)).numpy(),
             **{k: v.numpy() for k, v in sc.items()})
    for jn, jnet in judges.items():
        with torch.no_grad():
            r[f"judge_{jn}_pred1"] = jnet(th1.to(DEV), a.to(DEV)).cpu().numpy(); r[f"judge_{jn}_pred0"] = jnet(th0.to(DEV), a.to(DEV)).cpu().numpy()
    if args.save_delta:
        r["delta"] = d.numpy().astype(np.float32)
    r.update(extra)
    records.setdefault(name, []).append(r)


for c_ in classes:
    mi = sets[f"{args.task}_{args.set}_c{c_}"][: args.n_per_class]
    if args.traj and not args.traj_all:
        mi = mi[: max(5, args.n_per_class // 10)]
    if len(mi) == 0:
        print(f"[{tag}] class {c_}: no eligible models, skipped", flush=True); continue
    th0 = torch.as_tensor(np.ascontiguousarray(TH[2 * M + mi])).to(DEV); a = torch.as_tensor(act_m[mi]).to(DEV); c = torch.full((len(mi),), c_, dtype=torch.long, device=DEV)
    tc = time.time()
    if is_sane or monitor is not None:   # SANE-W / SANE-Z / monitor: alignment permutation computed at theta0 and held fixed along the path
        import sane_zoo as sz
        _ref = net.theta_ref if is_sane else _mon_net.theta_ref
        with torch.no_grad():
            if getattr(net if is_sane else _mon_net, "aligned", True):
                P0, _ = sz.weight_match_restarts(th0, _ref, restarts=args.restarts, seed=c_)
            else:
                P0 = sz.identity_perms(len(th0), th0.device)
        if is_sane:
            net.set_perms(P0)
        if monitor is not None:
            _mon_net.set_perms(P0)
    if kind in ("meta", "meta_raw", "meta_out"):
        kw = dict(eta=args.eta, max_steps=args.max_steps, step_mode="raw" if kind == "meta_raw" else "rel",
                  grad_mask=ec.output_layer_mask() if kind == "meta_out" else None)
        if args.traj or args.traj_all:
            kw.update(Xte=Xte_sub, yte=yte_sub, log_every=args.log_every, monitor=monitor)
        out = ec.edit(th0, a, c, psi, task=args.task, **kw)
        extra = dict(steps=out["steps"], stopped=out["stopped"], pred0=out["pred0"], pred1=out["pred_final"], stop_target=out["target"])
        if args.traj or args.traj_all:
            extra.update(traj_pred=out["traj_pred"], traj_true=out["traj_true"], traj_step=np.tile(out["traj_step"], (len(mi), 1)))
            if out.get("traj_mon") is not None:
                extra["traj_recon"] = out["traj_mon"]
        add(kind, mi, c, th0, out["theta"], a, extra)
        if kind == "meta" and not (args.traj and not args.traj_all) and not args.no_derived:
            d = out["theta"] - th0
            thr = ec.random_matched(th0, d, gen)
            with torch.no_grad():
                pr = psi(thr, a)
            add("random_matched", mi, c, th0, thr, a, dict(pred0=out["pred0"], pred1=pr, steps=out["steps"], stopped=out["stopped"]))
            thb = ec.bias_only_matched(th0, d, c, args.task)
            with torch.no_grad():
                pb = psi(thb, a)
            add("bias_matched", mi, c, th0, thb, a, dict(pred0=out["pred0"], pred1=pb, steps=out["steps"], stopped=out["stopped"]))
            # bias-only direction with the metanet as the stopping judge
            skip_bl = os.path.exists("job_overrides.json") and any(p in args.method for p in json.load(open("job_overrides.json")).get("skip_bias_line_for", []))
            if skip_bl:
                outb = None
            else:
                outb = ec.edit(th0, a, c, psi, task=args.task, eta=args.eta, max_steps=args.max_steps, direction=ec.bias_direction(c))
            if outb is not None:
                add("bias_line", mi, c, th0, outb["theta"], a, dict(steps=outb["steps"], stopped=outb["stopped"], pred0=outb["pred0"], pred1=outb["pred_final"]))
    elif kind in ("zedit", "zedit_dense", "zdec"):
        kw = dict(eta=args.eta_z, max_steps=args.max_steps, dense_only=(kind == "zedit_dense"), gen=gen,
                  zrandom=not args.no_derived)
        if kind == "zdec":
            if args.zpsi:
                _zp = ec.load_metanet(f"runs/{args.zpsi}.pt"); kw["psi_theta"] = lambda th, a_: _zp(th, a_)
            else:
                kw["psi_theta"] = lambda th, a_: net(th, a_)      # SANE round trip (encoder + head, perms fixed at theta0)
        if args.traj or args.traj_all:
            kw.update(Xte=Xte_sub, yte=yte_sub, log_every=args.log_every, monitor=monitor)
        out = ec.edit_z(th0, a, c, net, task=args.task, **kw)
        extra = dict(steps=out["steps"], stopped=out["stopped"], pred0=out["pred0"], pred1=out["pred_final"], stop_target=out["target"],
                     z_delta_norm=out["z_delta_norm"], z0_norm=out["z0_norm"])
        if args.traj or args.traj_all:
            extra.update(traj_pred=out["traj_pred"], traj_true=out["traj_true"], traj_step=np.tile(out["traj_step"], (len(mi), 1)),
                         traj_recon=out["traj_mon"])
        rname = kind + (f"_{args.zpsi.split('_s')[0]}" if (kind == "zdec" and args.zpsi) else "")
        add(rname, mi, c, th0, out["theta"], a, extra)
        if not args.no_derived:
            if kind == "zedit":
                add("sane_identity", mi, c, th0, out["theta_identity"], a, dict(pred0=out["pred0"], pred1=out["pred_identity"],
                    steps=torch.zeros_like(out["steps"]), stopped=torch.zeros_like(out["stopped"])))
            add("zrandom" if kind == "zedit" else f"{rname}_zrandom", mi, c, th0, out["theta_zrandom"], a,
                dict(pred0=out["pred0"], pred1=out["pred_zrandom"], steps=out["steps"], stopped=out["stopped"], z_delta_norm=out["z_delta_norm"]))
    elif kind == "ga_probes":
        # pseudo-data gradient instead of metanet gradient; same normalised step and same metanet stop
        sign = -1 if args.task == "unlearn" else +1       # CE descent (+1) learns c; ascent (-1) unlearns c
        Xp, yp = ec.probe_pseudo_data(net, th0, a, c)
        theta = th0.clone(); norm0 = th0.norm(dim=1, keepdim=True)
        with torch.no_grad():
            pred0 = psi(th0, a)
        p0 = pred0.gather(1, c[:, None])[:, 0]
        target = p0 * 0.4 if args.task == "unlearn" else torch.full_like(p0, 0.9)
        active = torch.ones(len(mi), dtype=torch.bool); steps = torch.zeros(len(mi), dtype=torch.long)
        for t in range(args.max_steps + 1):
            with torch.no_grad():
                pc = psi(theta, a).gather(1, c[:, None])[:, 0]
            done = (pc <= target) if args.task == "unlearn" else (pc >= target)
            active = active & ~done
            if t == args.max_steps or not active.any():
                break
            th = theta.detach().requires_grad_(True)
            logits = cnnzoo.forward(th, Xp, a, shared_x=False)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 10), yp.reshape(-1), reduction="sum")
            g, = torch.autograd.grad(loss, th)
            g = sign * g
            with torch.no_grad():
                g = g / (g.norm(dim=1, keepdim=True) + 1e-12) * norm0
                theta = theta - args.eta * g * active[:, None].float()
            steps += active.long()
        with torch.no_grad():
            pred1 = psi(theta, a)
        add(kind, mi, c, th0, theta, a, dict(steps=steps, stopped=(~active), pred0=pred0, pred1=pred1))
    elif kind in ("ga_real", "ft_retain"):
        rng = np.random.default_rng(args.seed + c_)
        n = 256
        # held-out real images of class c (MNIST-train, disjoint from the fitting images) for the data-based stop rule
        pool_all = np.where(ytr == c_)[0]
        hold = rng.choice(pool_all, 200, replace=False)            # held-out class-c images for the data-based stop rule
        pool_c = np.setdiff1d(pool_all, hold)
        if kind == "ga_real":
            idx = np.stack([rng.choice(pool_c, n, replace=False) for _ in range(len(mi))])
            sign, steps_n = (-1 if args.task == "unlearn" else +1), 200
        else:
            if args.task == "unlearn":
                idx = np.stack([rng.choice(np.where(ytr != c_)[0], n, replace=False) for _ in range(len(mi))])
            else:
                idx = np.stack([np.concatenate([rng.choice(pool_c, n // 2, replace=False),
                                                rng.choice(np.where(ytr != c_)[0], n // 2, replace=False)]) for _ in range(len(mi))])
            sign, steps_n = +1, 100
        X = cnnzoo.prep_images(Xtr_u8[idx.reshape(-1)]).reshape(len(mi), n, 1, *IMG_HW).to(DEV)
        y = torch.as_tensor(ytr[idx].astype(np.int64), device=DEV)
        lr = args.lr_mult * th0.pow(2).mean(1).sqrt()
        stop_fn = None
        if kind == "ga_real":
            Xh = cnnzoo.prep_images(Xtr_u8[hold]).to(DEV)
            with torch.no_grad():
                r0 = (cnnzoo.forward(th0, Xh, a).argmax(-1) == c_).float().mean(1)
            def stop_fn(th):
                rc = (cnnzoo.forward(th, Xh, a).argmax(-1) == c_).float().mean(1)
                return rc <= 0.4 * r0 if args.task == "unlearn" else rc >= 0.9
        th1, used, stopped = ec.finetune(th0, a, X, y, lr, steps_n, sign=sign, gen=gen, stop_fn=stop_fn)
        add(kind, mi, c, th0, th1, a, dict(steps=used, stopped=stopped))
    else:
        raise SystemExit(f"unknown method {kind}")
    print(f"[{tag}] class {c_} n={len(mi)} done in {time.time()-tc:.0f}s", flush=True)

# ---------------- save (concatenate per method)
out = {}
def _pad(arrs):
    w = max(a.shape[1] for a in arrs)
    return [np.pad(a.astype(np.float64), ((0, 0), (0, w - a.shape[1])), constant_values=np.nan) if a.shape[1] < w else a for a in arrs]


for name, recs in records.items():
    for k in recs[0]:
        if np.ndim(recs[0][k]) <= 1:
            out[f"{name}__{k}"] = np.concatenate([np.atleast_1d(r[k]) for r in recs])
        else:
            arrs = [r[k] for r in recs]
            if len({a.shape[1] for a in arrs}) > 1:   # ragged trajectories -> NaN-pad
                arrs = _pad(arrs)
            out[f"{name}__{k}"] = np.concatenate(arrs)
meta_out = dict(method=args.method, task=args.task, set=args.set, eta=args.eta, eta_z=args.eta_z, max_steps=args.max_steps, n_per_class=args.n_per_class,
                judges=list(judges), lr_mult=args.lr_mult, seed=args.seed, sane_monitor=args.sane_monitor, time_s=time.time() - t0)
np.savez(f"results/{tag}.npz", meta=json.dumps(meta_out), **out)
print(f"[{tag}] DONE {time.time()-t0:.0f}s", flush=True)

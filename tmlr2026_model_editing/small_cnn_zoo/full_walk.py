"""Full-length Algorithm-1 unlearning walk on the Small-CNN MNIST zoo: NO stop of any kind.
Same update as edit_cnn.edit / free_walk.py (Boost loss beta=0.1 on psi, gradient optionally masked to the output layer,
normalised step theta <- theta - eta*||theta0||*g/||g||), but every edit walks the full --cap steps (default 2000), as
ublib.walk does on the blob zoo. Both stop rules are only FLAGGED:
  psi-stop : first step t (checked every step, before the update, as edit_cnn.edit) with psi_c(theta_t) <= 0.4*psi_c(theta_0);
             psi_c and the true recall_c (2,000-image subset) of that state are stored.
  oracle   : first logged step (every --log_every steps) with true recall_c on the fixed stratified 2,000-image test subset
             (rng 12345, 200/class, as run_edit.py --traj_sub / free_walk.py) <= 0.4 x its step-0 value.
psi_c and true recall_c of EVERY edit are logged at every --log_every steps from 0 to --cap.
SANE editors: alignment perms from weight_match_restarts on the full class batch (restarts 8, seed=class), held fixed.
Resumable: checkpoint to <state_dir>/state_c<c>.pt at a log boundary when the wall budget is exhausted.
usage (cwd = code dir): python full_walk.py --edits edits.npz --run graph1_s0_e30 --key meta --state_dir DIR --classes 0,1 [--budget_s 780]"""
import os, sys, json, time, argparse
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import cnnzoo, edit_cnn as ec

ap = argparse.ArgumentParser()
ap.add_argument("--edits", required=True); ap.add_argument("--run", required=True); ap.add_argument("--key", default="meta")
ap.add_argument("--state_dir", required=True); ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--eta", type=float, default=1e-3); ap.add_argument("--cap", type=int, default=2000); ap.add_argument("--log_every", type=int, default=20)
ap.add_argument("--budget_s", type=float, default=1e12); ap.add_argument("--restarts", type=int, default=8); ap.add_argument("--max_models", type=int, default=0)
args = ap.parse_args()
T0 = time.time(); DEV = cnnzoo.DEV; os.makedirs(args.state_dir, exist_ok=True)
E = np.load(args.edits)
meta = np.load("zoo_meta.npz", allow_pickle=True); TH = np.load("zoo_theta.npy", mmap_mode="r"); M = len(meta["model"])
_, _, Xte_u8, yte = cnnzoo.load_images("mnist")
Xte = cnnzoo.prep_images(Xte_u8).to(DEV); yte_t = torch.as_tensor(yte.astype(np.int64), device=DEV)
_g = np.random.default_rng(12345)
_sub = np.concatenate([_g.choice(np.where(yte == k)[0], min(200, int((yte == k).sum())), replace=False) for k in range(10)])
Xs, ys = Xte[_sub], yte_t[_sub]
run = args.run
net = ec.load_metanet(f"runs/{run.split('@')[0]}.pt" + (("@" + run.split("@")[1]) if "@" in run else ""))
is_sane = hasattr(net, "set_perms")
if is_sane:
    net.restarts = args.restarts
    import sane_zoo as sz
mask = ec.output_layer_mask() if args.key == "meta_out" else None
LE = args.log_every; NLOG = args.cap // LE + 1


def out_of_time():
    return time.time() - T0 > args.budget_s


status = "complete"
for c_ in [int(c) for c in args.classes.split(",")]:
    resf = os.path.join(args.state_dir, f"res_c{c_}.npz"); stf = os.path.join(args.state_dir, f"state_c{c_}.pt")
    if os.path.exists(resf):
        print(f"class {c_}: done already", flush=True); continue
    if out_of_time():
        status = "incomplete"; break
    rows = np.where(E["c"] == c_)[0]
    if args.max_models:
        rows = rows[: args.max_models]
    mi = E["model"][rows]; B = len(rows); ar = torch.arange(B, device=DEV)
    th0 = torch.as_tensor(np.ascontiguousarray(TH[2 * M + mi])).to(DEV)
    act = torch.as_tensor(E["act"][rows].astype(np.int64)).to(DEV); cc = torch.full((B,), c_, dtype=torch.long, device=DEV)
    norm0 = th0.norm(dim=1, keepdim=True)
    if os.path.exists(stf):
        S = torch.load(stf, map_location=DEV, weights_only=False)
        theta, t, p0, P0, init_sub = S["theta"], S["t"], S["p0"], S["P0"], S["init_sub"]
        s_psi, s_or, psi_stop, true_stop, Lt, Lp, wall_prev = S["s_psi"], S["s_or"], S["psi_stop"], S["true_stop"], S["Lt"], S["Lp"], S["wall"]
        print(f"class {c_}: resumed at t={t}", flush=True)
    else:
        P0 = None
        if is_sane:
            with torch.no_grad():
                P0 = sz.weight_match_restarts(th0, net.theta_ref, restarts=args.restarts, seed=c_)[0] if getattr(net, "aligned", True) \
                     else sz.identity_perms(B, th0.device)
            net.set_perms(P0)
        with torch.no_grad():
            p0 = net(th0, act).gather(1, cc[:, None])[:, 0]
        theta = th0.clone(); t = 0; init_sub = None; wall_prev = 0.0
        s_psi = torch.full((B,), -1, dtype=torch.long, device=DEV); s_or = torch.full((B,), -1, dtype=torch.long, device=DEV)
        psi_stop = torch.full((B,), float("nan"), device=DEV); true_stop = torch.full((B,), float("nan"), device=DEV)
        Lt = torch.full((B, NLOG), float("nan"), device=DEV); Lp = torch.full((B, NLOG), float("nan"), device=DEV)
    target = 0.4 * p0
    tc = time.time()
    while True:
        if t % LE == 0 and t > 0 and t < args.cap and out_of_time():
            torch.save(dict(theta=theta, t=t, p0=p0, P0=P0, init_sub=init_sub, s_psi=s_psi, s_or=s_or, psi_stop=psi_stop, true_stop=true_stop,
                            Lt=Lt, Lp=Lp, wall=wall_prev + time.time() - tc), stf + ".tmp")
            os.replace(stf + ".tmp", stf); status = "incomplete"
            print(f"class {c_}: checkpoint at t={t}", flush=True); break
        th = theta.detach().requires_grad_(True)
        if is_sane:
            net.set_perms(P0)
        pred = net(th, act)
        pc = pred.gather(1, cc[:, None])[:, 0].detach()
        newly = (s_psi < 0) & (pc <= target)                      # psi-stop flag (state BEFORE the update, as edit_cnn.edit)
        if newly.any():
            n_idx = torch.where(newly)[0]
            with torch.no_grad():
                true_stop[n_idx] = ec.true_recall(theta[n_idx], act[n_idx], Xs, ys).gather(1, cc[n_idx][:, None])[:, 0]
            psi_stop[n_idx] = pc[n_idx]; s_psi[n_idx] = t
        if t % LE == 0:
            with torch.no_grad():
                rec = ec.true_recall(theta, act, Xs, ys).gather(1, cc[:, None])[:, 0]
            if t == 0:
                init_sub = rec.clone()
            Lt[:, t // LE] = rec; Lp[:, t // LE] = pc
            fire = (s_or < 0) & (rec <= 0.4 * init_sub)
            s_or[fire] = t
        if t >= args.cap:
            break
        f = ec.loss_fn(pred, cc, "boost", 0.1, +1).sum()
        g, = torch.autograd.grad(f, th)
        if mask is not None:
            g = g * mask[None, :]
        with torch.no_grad():
            g = g / (g.norm(dim=1, keepdim=True) + 1e-12) * norm0
            theta = theta - args.eta * g
        t += 1
    if status == "incomplete":
        break
    with torch.no_grad():
        r1 = ec.true_recall(theta, act, Xte, yte_t)
    np.savez(resf + ".tmp.npz", idx=rows, model=mi, c=cc.cpu().numpy(), act=act.cpu().numpy(), pred0=p0.cpu().numpy(), init_sub=init_sub.cpu().numpy(),
             s_psi=s_psi.cpu().numpy(), s_or=s_or.cpu().numpy(), psi_stop=psi_stop.cpu().numpy(), true_stop=true_stop.cpu().numpy(),
             log_true=Lt.cpu().numpy().astype(np.float32), log_psi=Lp.cpu().numpy().astype(np.float32), rec_final_fulltest=r1.cpu().numpy(),
             delta_norm_final=(theta - th0).norm(dim=1).cpu().numpy(), theta0_norm=norm0[:, 0].cpu().numpy(),
             log_every=LE, cap=args.cap, wall_s=wall_prev + time.time() - tc)
    os.replace(resf + ".tmp.npz", resf)
    if os.path.exists(stf):
        os.remove(stf)
    sp = s_psi.cpu().numpy(); so = s_or.cpu().numpy()
    print(f"class {c_}: DONE n={B} psi_stopped={int((sp >= 0).sum())} median_psi_step={np.median(sp[sp >= 0]) if (sp >= 0).any() else -1} "
          f"oracle_met={int((so >= 0).sum())} median_oracle_step={np.median(so[so >= 0]) if (so >= 0).any() else -1} wall={wall_prev + time.time() - tc:.0f}s", flush=True)
json.dump(dict(status=status, classes=args.classes, run=run, key=args.key, wall=time.time() - T0), open(os.path.join(args.state_dir, f"status_{args.classes.replace(',', '')}.json"), "w"))
print(f"STATUS {status} ({time.time()-T0:.0f}s)", flush=True)

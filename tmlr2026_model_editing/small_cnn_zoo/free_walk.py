"""Free-walk oracle unlearning edit (no psi-stop).
Algorithm 1 update exactly as edit_cnn.edit (Boost loss beta=0.1 on psi, gradient optionally masked to the output layer,
normalised step theta <- theta - eta*||theta0||*g/||g||), but the psi-stop is NOT applied: every edit keeps walking until
the ORACLE fires, i.e. the first logged step (every --log_every steps, incl. step 0) at which TRUE recall_c on the fixed
stratified 2,000-image test subset (rng 12345, 200/class, as run_edit.py --traj_sub) is <= 0.4 x its step-0 value. The
edit state is the state at that step. Safety cap --cap steps: edits still running at the cap are 'not reached' and keep
their cap state. SANE editors: alignment perms from weight_match_restarts on the full class batch (restarts 8, seed=class),
held fixed, as run_edit.py. Only still-running rows are stepped (identical to the frozen-row update of edit_cnn.edit).
Resumable: state is checkpointed to <state_dir>/state_c<c>.pt at a log boundary when the wall budget is exhausted;
re-running the same command resumes. Finished classes -> <state_dir>/res_c<c>.npz.
usage: python free_walk.py --edits edits.npz --run graph1_s0_e30 --key meta --state_dir DIR --classes 0,1 [--budget_s 780]"""
import os, sys, json, time, argparse
sys.path.append(os.getcwd())
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
import cnnzoo, edit_cnn as ec

ap = argparse.ArgumentParser()
ap.add_argument("--edits", required=True); ap.add_argument("--run", required=True); ap.add_argument("--key", default="meta")
ap.add_argument("--state_dir", required=True); ap.add_argument("--classes", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--eta", type=float, default=1e-3); ap.add_argument("--cap", type=int, default=10000); ap.add_argument("--log_every", type=int, default=20)
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


def psi_c(theta, act, cc, rows_idx, P0):
    if is_sane:
        net.set_perms(P0[rows_idx])
    return net(theta, act).gather(1, cc[:, None])[:, 0]


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
    mi = E["model"][rows]; B = len(rows)
    th0 = torch.as_tensor(np.ascontiguousarray(TH[2 * M + mi])).to(DEV)
    act = torch.as_tensor(E["act"][rows].astype(np.int64)).to(DEV); cc = torch.full((B,), c_, dtype=torch.long, device=DEV)
    norm0 = th0.norm(dim=1, keepdim=True)
    if os.path.exists(stf):
        S = torch.load(stf, map_location=DEV, weights_only=False)
        theta, active, reached, s_or, steps, t = S["theta"], S["active"], S["reached"], S["s_or"], S["steps"], S["t"]
        init_sub, p0, P0 = S["init_sub"], S["p0"], S["P0"]; Lt, Lp = S["Lt"], S["Lp"]; wall_prev = S["wall"]
        print(f"class {c_}: resumed at t={t}, active={int(active.sum())}", flush=True)
    else:
        P0 = None
        if is_sane:
            with torch.no_grad():
                P0 = sz.weight_match_restarts(th0, net.theta_ref, restarts=args.restarts, seed=c_)[0] if getattr(net, "aligned", True) \
                     else sz.identity_perms(B, th0.device)
        with torch.no_grad():
            p0 = psi_c(th0, act, cc, torch.arange(B, device=DEV), P0)
        theta = th0.clone(); active = torch.ones(B, dtype=torch.bool, device=DEV); reached = torch.zeros(B, dtype=torch.bool, device=DEV)
        s_or = torch.full((B,), -1, dtype=torch.long, device=DEV); steps = torch.zeros(B, dtype=torch.long, device=DEV); t = 0
        init_sub = None; Lt = torch.full((B, NLOG), float("nan"), device=DEV); Lp = torch.full((B, NLOG), float("nan"), device=DEV); wall_prev = 0.0
    tc = time.time()
    while True:
        if t % LE == 0:
            if t > 0 and out_of_time():
                torch.save(dict(theta=theta, active=active, reached=reached, s_or=s_or, steps=steps, t=t, init_sub=init_sub, p0=p0, P0=P0,
                                Lt=Lt, Lp=Lp, wall=wall_prev + time.time() - tc), stf + ".tmp")
                os.replace(stf + ".tmp", stf); status = "incomplete"
                print(f"class {c_}: checkpoint at t={t}, active={int(active.sum())}, reached={int(reached.sum())}", flush=True); break
            a_idx = torch.where(active)[0]
            with torch.no_grad():
                rec = ec.true_recall(theta[a_idx], act[a_idx], Xs, ys).gather(1, cc[a_idx][:, None])[:, 0]
                pc = psi_c(theta[a_idx], act[a_idx], cc[a_idx], a_idx, P0)
            if t == 0:
                init_sub = rec.clone()
            Lt[a_idx, t // LE] = rec; Lp[a_idx, t // LE] = pc
            fire = rec <= 0.4 * init_sub[a_idx]
            reached[a_idx[fire]] = True; s_or[a_idx[fire]] = t; active[a_idx[fire]] = False
            if t >= args.cap:
                active[:] = False
        if not active.any():
            break
        a_idx = torch.where(active)[0]
        th = theta[a_idx].detach().requires_grad_(True)
        if is_sane:
            net.set_perms(P0[a_idx])
        pred = net(th, act[a_idx])
        f = ec.loss_fn(pred, cc[a_idx], "boost", 0.1, +1).sum()
        g, = torch.autograd.grad(f, th)
        if mask is not None:
            g = g * mask[None, :]
        with torch.no_grad():
            g = g / (g.norm(dim=1, keepdim=True) + 1e-12) * norm0[a_idx]
            theta[a_idx] = theta[a_idx] - args.eta * g
        steps[a_idx] += 1; t += 1
    if status == "incomplete":
        break
    with torch.no_grad():
        p1 = psi_c(theta, act, cc, torch.arange(B, device=DEV), P0)
        r1 = ec.true_recall(theta, act, Xte, yte_t)
    delta = (theta - th0)
    np.savez(resf + ".tmp.npz", idx=rows, model=mi, c=cc.cpu().numpy(), act=act.cpu().numpy(), reached=reached.cpu().numpy(), s_or=s_or.cpu().numpy(),
             steps=steps.cpu().numpy(), pred0=p0.cpu().numpy(), pred1=p1.cpu().numpy(), rec1=r1.cpu().numpy(), init_sub=init_sub.cpu().numpy(),
             delta=delta.cpu().numpy().astype(np.float32), delta_norm=delta.norm(dim=1).cpu().numpy(), log_true=Lt.cpu().numpy().astype(np.float16),
             log_psi=Lp.cpu().numpy().astype(np.float16), log_every=LE, cap=args.cap, wall_s=wall_prev + time.time() - tc)
    os.replace(resf + ".tmp.npz", resf)
    if os.path.exists(stf):
        os.remove(stf)
    st = steps.cpu().numpy(); rr = reached.cpu().numpy()
    print(f"class {c_}: DONE n={B} reached={int(rr.sum())} cap_hits={int((~rr).sum())} median_step={np.median(s_or.cpu().numpy()[rr]) if rr.any() else -1} "
          f">2000={int((st > 2000).sum())} wall={wall_prev + time.time() - tc:.0f}s", flush=True)
json.dump(dict(status=status, classes=args.classes, run=run, key=args.key, wall=time.time() - T0), open(os.path.join(args.state_dir, f"status_{args.classes.replace(',', '')}.json"), "w"))
print(f"STATUS {status} ({time.time()-T0:.0f}s)", flush=True)

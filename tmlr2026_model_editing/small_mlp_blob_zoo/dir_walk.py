"""Per-class edit directions on the blobs zoo: every selected model is edited toward EVERY class.
usage: python dir_walk.py <spec mlp_raw|neuron_r1|sane> <seed 0|1> [n_models=300]
Protocol identical to ublib.walk (Fig 2): Boost beta=0.1, normalised step eta*||theta0||, psi-stop
psi_c <= 0.4 psi_c(theta0), oracle = first logged step (every 20) with probe recall_c <= 0.4 initial, cap 2000.
Walk exits early once every row has psi-stopped, met the oracle and passed step 100.
Rows are model-major: row i = model i//K, class i%K. Saves direction/edit vectors (float32)."""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.getcwd())
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEV)
import ublib as U
from toyzoo import per_class_recall, K, forward_flat
import toyzoo as T

spec, seed = sys.argv[1], int(sys.argv[2]); NM = int(sys.argv[3]) if len(sys.argv) > 3 else 300
z = U.load_zoo()
te = np.where(z["split"] == "test")[0]; R = z["recall"][-1, te]
ok = np.where(R.min(1) >= 0.5)[0]
rng = np.random.default_rng(2026)
Msel = te[np.sort(rng.choice(ok, NM, replace=False))]
M = np.repeat(Msel, K); C = np.tile(np.arange(K), NM)
if spec == "sane":
    torch.set_default_device("cpu")
    import sane_psi as SP
    psi = SP.SanePsi(U.SANE_PT[seed], M)
    SP.MASK = SP.MASK.to(DEV); SP.POS = SP.POS.to(DEV)
    torch.set_default_device(DEV)
    psi.ae.to(DEV); psi.head.to(DEV); psi.P = psi.P.to(DEV); psi.mu = psi.mu.to(DEV); psi.sg = psi.sg.to(DEV)
else:
    psi = U.get_psi(spec, seed, z, M)
assert psi is not None
# SANE seed-0 git-re-basin permutations for the selected models (for aligned cross-model comparison)
dsp = torch.load(U.SANE_PT[0], map_location="cpu", weights_only=False)
perm = dsp["perm"][torch.as_tensor(Msel, device="cpu")].cpu().numpy()

theta0 = torch.tensor(z["theta"][-1, M], dtype=torch.float32); c = torch.tensor(C)
Xp, yp = U.probe_set(); Xe, ye = U.eval_set()
B = theta0.shape[0]; ar = torch.arange(B); n0 = theta0.norm(dim=1); t_start = time.time()
with torch.no_grad():
    p0 = psi(theta0)[ar, c]
r0p, _ = per_class_recall(theta0, Xp, yp); t0 = r0p[ar, c]
oth = torch.ones(B, K); oth[ar, c] = 0
theta = theta0.clone()
fired = torch.zeros(B, dtype=torch.bool); fstep = torch.full((B,), U.MAXS); th_ms = theta0.clone()
or_met = torch.zeros(B, dtype=torch.bool); or_step = torch.full((B,), -1); th_or = theta0.clone()
best = t0.clone(); th_best = theta0.clone(); best_step = torch.zeros(B, dtype=torch.long)
SNAP = (20, 100); snaps = {}; g0 = None; g_ms = torch.zeros_like(theta0)
for t in range(U.MAXS + 1):
    th = theta.detach().requires_grad_(t < U.MAXS)
    with torch.set_grad_enabled(t < U.MAXS):
        P = psi(th)
    pc = P[ar, c].detach()
    nw = (~fired) & (pc <= U.REL * p0); fired |= nw; fstep[nw] = t; th_ms[nw] = theta[nw]
    if t % U.LOG == 0:
        rp, _ = per_class_recall(theta, Xp, yp); tc = rp[ar, c]
        hit = (~or_met) & (tc <= U.REL * t0); or_met |= hit; or_step[hit] = t; th_or[hit] = theta[hit]
        imp = tc < best; best[imp] = tc[imp]; th_best[imp] = theta[imp]; best_step[imp] = t
    if t in SNAP:
        snaps[t] = theta.clone()
    if t == U.MAXS or (t > max(SNAP) and bool(fired.all()) and bool(or_met.all())):
        break
    f = (P[ar, c] - U.BETA * (P * oth).sum(1)).sum()
    g, = torch.autograd.grad(f, th)
    gn = g.norm(dim=1, keepdim=True)
    u = torch.where(gn > 0, g / gn.clamp_min(1e-12), torch.zeros_like(g))
    if t == 0:
        g0 = -u.clone()                      # initial edit direction (unit)
    theta = (theta - u * (U.ETA * n0)[:, None]).detach()
t_end = t
th_ms[~fired] = theta[~fired]
th_orf = torch.where(or_met[:, None], th_or, th_best); or_stepf = torch.where(or_met, or_step, best_step)
for s in SNAP:
    snaps.setdefault(s, theta.clone())

# data-gradient references on the SAME models: unlearn-ascent direction +grad CE(class-k data), all k
drng = np.random.default_rng(7)
Xd = torch.tensor(np.stack([T.sample_class(drng, k, 256) for k in range(K)]))  # (K,256,2)
th0m = torch.tensor(z["theta"][-1, Msel], dtype=torch.float32)
gdata = torch.zeros(NM, K, th0m.shape[1])
for k in range(K):
    th = th0m.clone().requires_grad_(True)
    lo = forward_flat(th, Xd[k])
    L = torch.nn.functional.cross_entropy(lo.reshape(-1, K), torch.full((NM * 256,), k), reduction="none").reshape(NM, -1).mean(1).sum()
    gk, = torch.autograd.grad(L, th); gdata[:, k] = gk

rec = lambda x: per_class_recall(x, Xe, ye)[0]
cpu = lambda x: x.detach().cpu().numpy().astype(np.float32)
out = dict(models=Msel, rows_model=M, rows_cls=C, theta0=cpu(th0m), perm=perm,
           g0=cpu(g0), d_psi=cpu(th_ms - theta0), d_oracle=cpu(th_orf - theta0),
           d_s20=cpu(snaps[20] - theta0), d_s100=cpu(snaps[100] - theta0),
           step_psi=cpu(fstep), fired=cpu(fired), step_oracle=cpu(or_stepf), oracle_met=cpu(or_met),
           rec0=cpu(rec(theta0)), rec_psi=cpu(rec(th_ms)), rec_oracle=cpu(rec(th_orf)), rec_s100=cpu(rec(snaps[100])),
           p0=cpu(p0), gdata=cpu(gdata), t_end=t_end, spec=spec, seed=seed)
os.makedirs("out", exist_ok=True)
np.savez_compressed(f"out/dir_{spec}_s{seed}.npz", **out)
print(spec, seed, DEV, f"B={B} steps={t_end} {time.time()-t_start:.0f}s psi-fired {float(fired.float().mean()):.3f} "
      f"oracle {float(or_met.float().mean()):.3f} median psi step {float(fstep.float().median()):.0f}", flush=True)

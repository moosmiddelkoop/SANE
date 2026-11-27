import os, sys, time, numpy as np, torch; sys.path.insert(0, os.getcwd()); torch.set_num_threads(4)
import ublib as U; from sane_psi import SanePsi
z = U.load_zoo(); jd = sys.argv[1]
D = np.load(f"{jd}/out/preds_test.npz"); fin = D["ckpt"] == 2
mods = D["model"][fin][:2000]; ref = D["sane_token"][fin][:2000]
psi = SanePsi(f"{jd}/out/sane.pt", mods)
with torch.no_grad(): P = psi(torch.tensor(z["theta"][-1, mods])).numpy()
print("CPU fp32 vs GPU bf16 stored token-head preds: max|diff|", float(np.abs(P - ref).max()), "median", float(np.median(np.abs(P - ref))))
M, C = U.edit_set(z); psi = SanePsi(f"{jd}/out/sane.pt", M); th = torch.tensor(z["theta"][-1, M]).requires_grad_(True)
t = time.time(); p = psi(th); g, = torch.autograd.grad(p[torch.arange(len(M)), torch.tensor(C)].sum(), th); print("fwd+bwd 1200 models", round(time.time() - t, 2), "s; grad norm median", float(g.norm(dim=1).median()))

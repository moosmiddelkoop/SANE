"""GA* (data-based stop) and retain FT* (fixed 100 steps) on the unified blobs edit set."""
import os, sys, time, numpy as np, torch; sys.path.insert(0, os.getcwd())
torch.set_num_threads(int(os.environ.get("NT", "1")))
import ublib as U
from toyzoo import forward_flat
z = U.load_zoo(); M, C = U.edit_set(z); B = len(M); ar = torch.arange(B)
th0 = torch.tensor(z["theta"][-1, M]); c = torch.tensor(C)
rng = np.random.default_rng(7); gen = torch.Generator().manual_seed(7); t0 = time.time()
# GA*: Adam ascent on 256 real class-c samples, lr = 0.01*rms(theta0), batch 64, <=200 steps,
# data stop: recall_c on 200 held-out class-c samples <= 0.4 x its start value (checked before each step)
Xf, yf = U.class_data(rng, C, 256); Xh, yh = U.class_data(rng, C, 200)
def rec_ho(th):
    with torch.no_grad(): return (forward_flat(th, Xh).argmax(-1) == yh).float().mean(1)
h0 = rec_ho(th0)
lr_ga = 0.01 * th0.pow(2).mean(1).sqrt()
ga, ga_steps, ga_stop = U.adam_ft(th0, Xf, yf, lr_ga, 200, sign=-1, gen=gen, stop_fn=lambda th: rec_ho(th) <= U.REL * h0)
print("GA* done", round(time.time() - t0), "stopped", float(ga_stop.float().mean()), "median steps", float(ga_steps.float().median()), flush=True)
# FT*: 100 Adam steps on 256 real retain samples (classes != c), lr = the model's own training lr (prior toy runs), batch 64
Xr, yr = U.class_data(rng, C, 256, exclude=True)
ft, ft_steps, _ = U.adam_ft(th0, Xr, yr, torch.tensor(z["lr"][M], dtype=torch.float32), 100, sign=+1, gen=gen)
print("FT* done", round(time.time() - t0), flush=True)
np.savez("ures/data_baselines.npz", model=M, cls=C, theta0=th0.numpy(), ga=ga.numpy(), ga_steps=ga_steps.numpy(), ga_stopped=ga_stop.numpy(),
         ft=ft.numpy(), ft_steps=ft_steps.numpy(), lr_ga=lr_ga.numpy())

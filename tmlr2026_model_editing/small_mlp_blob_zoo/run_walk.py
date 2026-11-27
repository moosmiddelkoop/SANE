"""usage: python run_walk.py <spec mlp_raw|neuron_r1> <scope full|output> [max_steps]"""
import os, sys, time, numpy as np, torch; sys.path.insert(0, os.getcwd())
torch.set_num_threads(int(os.environ.get("NT", "1")))
import ublib as U
spec, scope = sys.argv[1], sys.argv[2]; ms = int(sys.argv[3]) if len(sys.argv) > 3 else U.MAXS
z = U.load_zoo(); M, C = U.edit_set(z)
th0 = torch.tensor(z["theta"][-1, M]); c = torch.tensor(C)
psi = U.get_psi(spec, 0, z, M)
Xp, yp = U.probe_set(); t0 = time.time()
r = U.walk(th0, c, psi, mask=U.out_mask() if scope == "output" else None, Xp=Xp, yp=yp, max_steps=ms)
tag = f"{spec}_{scope}"
np.save(f"ures/snaps_{tag}.npy", r.pop("snaps").numpy().astype(np.float32))
np.savez(f"ures/walk_{tag}.npz", model=M, cls=C, theta0=th0.numpy(), **{k: v.numpy() for k, v in r.items()})
print(tag, "done", round(time.time() - t0), "s; psi stopped", float(r["stopped"].float().mean()), "oracle met", float(r["oracle_met"].float().mean()), flush=True)

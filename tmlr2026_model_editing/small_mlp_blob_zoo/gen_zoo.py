import os, sys, time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("NT","3")))
import toyzoo as T
M, steps, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
t0 = time.time()
z = T.train_zoo(M, seed=777, steps=steps, n_train=256, batch=64, ckpt_fracs=(0.08, 0.3, 1.0), verbose=True)
perm = np.random.default_rng(0).permutation(M)
split = np.empty(M, dtype="<U5"); split[perm[:M // 2]] = "train"; split[perm[M // 2:3 * M // 4]] = "val"; split[perm[3 * M // 4:]] = "test"
z["split"] = split; z["task"] = np.array(T.TASK)
np.savez(out, **z)
print(T.TASK, "done", f"{time.time()-t0:.0f}s", flush=True)

"""Cluster staging: build zoo_theta.npy (3M x 4970, step-major 0/20/86) from the public Small-CNN-Zoo weights.npy
using the model list / order of <zoo>_meta.npz (built locally from the user's per-class labels), then verify
per-model norms against <zoo>_check.npz.   usage: python stage_zoo.py <zoo> <extracted_dir>"""
import sys, os, csv, numpy as np, time
zoo, src = sys.argv[1], sys.argv[2]
t0 = time.time()
zm = np.load(f"{zoo}_meta.npz", allow_pickle=False)
modeldir = zm["modeldir"]; M = len(modeldir)
# metrics.csv: find the row of each (modeldir, step)
pos = {m: i for i, m in enumerate(modeldir)}
rows = {s: np.full(M, -1, dtype=np.int64) for s in (0, 20, 86)}
with open(os.path.join(src, "metrics.csv"), newline="") as f:
    rd = csv.reader(f); hdr = next(rd); im = hdr.index("modeldir"); ist = hdr.index("step")
    for r, line in enumerate(rd):
        i = pos.get(line[im])
        if i is not None:
            s = int(line[ist])
            if s in rows:
                rows[s][i] = r
for s in rows:
    assert (rows[s] >= 0).all(), f"missing rows for step {s}"
assert (rows[86] == zm["row86"]).all(), "row86 mismatch with local metadata"
W = np.load(os.path.join(src, "weights.npy"), mmap_mode="r")
out = np.lib.format.open_memmap("zoo_theta.npy.tmp", mode="w+", dtype=np.float32, shape=(3 * M, 4970))
for i, s in enumerate((0, 20, 86)):
    order = np.argsort(rows[s])
    out[i * M + order] = W[rows[s][order]]
    print(f"step {s} copied ({time.time()-t0:.0f}s)", flush=True)
chk = np.load(f"{zoo}_check.npz")
norms = np.sqrt((out.astype(np.float64) ** 2).sum(1)).reshape(3, M)
rel = np.abs(norms - chk["norms"]) / (chk["norms"] + 1e-6)
print("norm check: max rel diff", rel.max(), "rows with diff>1e-5:", int((rel > 1e-5).sum()))
assert rel.max() < 1e-4, "theta rows do not match local zoo"
out.flush(); del out; os.replace("zoo_theta.npy.tmp", "zoo_theta.npy")
np.savez("zoo_meta.npz", **{k: zm[k] for k in zm.files})
print(zoo, "staged M", M, f"({time.time()-t0:.0f}s)")

"""Assemble zoo_theta.npy (3M x 4970, step-major 0/20/86) + zoo_meta.npz + edit_sets.npz for run_meta/run_edit.
usage: python prep_zoo.py <zoo> [src_dir]   -- reads <zoo>_theta{0,20,86}.npy + <zoo>_meta.npz + edit_sets_<zoo>.npz
   (or, if present, <zoo>_labelled.npz)"""
import sys, os, shutil, numpy as np
zoo = sys.argv[1]; src = sys.argv[2] if len(sys.argv) > 2 else "."
P = lambda n: os.path.join(src, n)
if os.path.exists(P(f"{zoo}_labelled.npz")):
    z = np.load(P(f"{zoo}_labelled.npz"), allow_pickle=False); parts = [z["theta"][i] for i in range(3)]
    names = np.array(["train", "val", "test"])
    meta = dict(model=np.arange(parts[0].shape[0]), recall=z["recall"], acc=z["acc"], act=z["act"], split=names[z["split"]],
                modeldir=z["modeldir"], row86=z["row86"], **{k: z[k] for k in z.files if k.startswith("cfg_")})
else:
    parts = [np.load(P(f"{zoo}_theta{s}.npy"), mmap_mode="r") for s in (0, 20, 86)]
    zm = np.load(P(f"{zoo}_meta.npz"), allow_pickle=False); meta = {k: zm[k] for k in zm.files}
M = parts[0].shape[0]
if not os.path.exists("zoo_theta.npy"):
    out = np.lib.format.open_memmap("zoo_theta.npy.tmp", mode="w+", dtype=np.float32, shape=(3 * M, 4970))
    for i in range(3):
        out[i * M:(i + 1) * M] = parts[i]
    out.flush(); del out; os.replace("zoo_theta.npy.tmp", "zoo_theta.npy")
np.savez("zoo_meta.npz", **meta)
shutil.copy(P(f"edit_sets_{zoo}.npz"), "edit_sets.npz")
print(zoo, "M", M, "theta", (3 * M, 4970), "split counts", {n: int((meta["split"] == n).sum()) for n in ("train", "val", "test")})

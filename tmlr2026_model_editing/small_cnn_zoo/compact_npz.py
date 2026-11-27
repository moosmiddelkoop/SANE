"""Rewrite run_edit result files without the per-edit edit vectors (__delta), for transfer.
usage: python compact_npz.py OUTDIR file1.npz [file2.npz ...]"""
import sys, os, numpy as np
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for f in sys.argv[2:]:
    z = np.load(f, allow_pickle=False)
    keep = {k: z[k] for k in z.files if not k.endswith("__delta")}
    np.savez_compressed(os.path.join(out, os.path.basename(f)), **keep)
    print(os.path.basename(f), len(z.files), "->", len(keep), "keys", os.path.getsize(os.path.join(out, os.path.basename(f))) // 1024, "KB")

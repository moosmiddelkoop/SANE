"""Drop the heavy per-edit delta arrays from result npz files. usage: python compact_results.py OUT_DIR file1.npz file2.npz ..."""
import sys, os, numpy as np
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for f in sys.argv[2:]:
    z = np.load(f, allow_pickle=True)
    keep = {k: z[k] for k in z.files if not (k.endswith("delta_fc16") or k.endswith("delta_full16"))}
    np.savez_compressed(os.path.join(out, os.path.basename(f)), **keep)
    print(os.path.basename(f), "->", os.path.getsize(os.path.join(out, os.path.basename(f))) // 1024, "KB", flush=True)

"""Edit sets from zoo_meta.npz: per class, unlearn = test models with recall_c >= 0.5, uplearn = 0.1 <= recall_c <= 0.8;
up to n per class (seed 0); val sets likewise from the val split; 'reduced' = classes 0,10,..,90 from the test sets.
usage: python make_edit_sets.py zoo_meta.npz edit_sets.npz [--n 8] [--n_val 5]"""
import sys, argparse, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("meta"); ap.add_argument("out"); ap.add_argument("--n", type=int, default=8); ap.add_argument("--n_val", type=int, default=5)
a = ap.parse_args(); m = np.load(a.meta, allow_pickle=True); rec = m["recall"]; split = m["split"]; rng = np.random.default_rng(0)
out = {}; stats = {}
for task in ("unlearn", "uplearn"):
    for sp, n in (("test", a.n), ("val", a.n_val)):
        idx = np.where(split == sp)[0]; M, C = [], []; elig = []
        for c in range(rec.shape[1]):
            ok = idx[(rec[idx, c] >= 0.5)] if task == "unlearn" else idx[(rec[idx, c] >= 0.1) & (rec[idx, c] <= 0.8)]
            elig.append(len(ok)); pick = rng.choice(ok, min(n, len(ok)), replace=False)
            M.append(pick); C.append(np.full(len(pick), c))
        out[f"{task}_{sp}_model"] = np.concatenate(M); out[f"{task}_{sp}_class"] = np.concatenate(C)
        stats[f"{task}_{sp}"] = (int(len(out[f"{task}_{sp}_model"])), int(min(elig)), int(np.median(elig)), int(sum(np.array(elig) < n)))
    keep = np.isin(out[f"{task}_test_class"], np.arange(0, 100, 10))
    out[f"{task}_reduced_model"] = out[f"{task}_test_model"][keep]; out[f"{task}_reduced_class"] = out[f"{task}_test_class"][keep]
np.savez(a.out, **out)
for k, v in stats.items(): print(k, "n_edits", v[0], "min eligible/class", v[1], "median eligible", v[2], "classes short of n", v[3])

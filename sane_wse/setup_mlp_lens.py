"""Set up and sanity-check one SANE MLP lens: split file, statistics, manifest, scratch index, scores."""
import json, random, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml

ds = sys.argv[1]
S = Path(sys.argv[2])
WSE = Path.home() / "Documents/WSL/wse"; SANE = Path.home() / "Documents/WSL/SANE"
ZOO = Path.home() / f"Documents/WSL/PyTorch-port/output/unthi_{ds}"
TRIAL = {"fmnist": "fmnist-v1.0_d3e0b_00000_2026-07-15_10-54-45", "cifar10": "cifar10-v1.0_d496b_00000_2026-07-15_10-54-47",
         "svhn": "svhn-v1.0_d4950_00000_2026-07-15_10-54-47", "mnist": "mnist-v2.0_6133f_00000_2026-07-15_11-48-49"}[ds]
out = SANE / "sane_pretraining" / ds
cache = torch.load(WSE / f"SANE_test_models/{ds}/mlp/embeddings_sorted.pt", map_location="cpu", weights_only=False)["8"]
mid, dirs = cache["mid"], cache["model_dirs"]
n = int(mid.max()) + 1; models = list(range(n)); random.Random(67).shuffle(models)
i1 = int(0.7 * n); i2 = i1 + int(0.15 * n)
by_mid = {int(m): d for m, d in zip(mid.tolist(), dirs)}
splits = {"train": sorted(by_mid[m] for m in models[:i1]), "val": sorted(by_mid[m] for m in models[i1:i2]), "test": sorted(by_mid[m] for m in models[i2:])}
(out / "splits_seed67.json").write_text(json.dumps(splits, indent=1))
print(ds, {k: len(v) for k, v in splits.items()}, flush=True)

keys = ["conv1.weight", "conv2.weight", "conv3.weight", "dense.weight"]
means = {k: [] for k in keys}; stds = {k: [] for k in keys}
t0 = time.time()
for d in splits["train"]:
    sd = torch.load(ZOO / d / "checkpoint_000008" / "checkpoints", map_location="cpu", weights_only=False)
    for k in keys:
        w = sd[k].view(sd[k].shape[0], -1); w = torch.cat([w, sd[k.replace("weight", "bias")].unsqueeze(1)], dim=1)
        means[k].append(w.mean().item()); stds[k].append(w.std().item())
stats = {k: {"mean": float(torch.tensor(means[k]).mean()), "std": float(torch.sqrt(torch.mean(torch.tensor(stds[k]) ** 2)))} for k in keys}
print(f"stats over {len(splits['train'])} train models in {time.time()-t0:.0f}s:", json.dumps(stats), flush=True)

manifest = SANE / f"sane_wse_manifest_{ds}_v1_mlp.yaml"
manifest.write_text(f"""# SANE encoder {TRIAL} (checkpoint_000050, fetched from Snellius 2026-09-29) + the MLP per-class
# recall head Moos trained on its embeddings (seed 67), for the wse unlearning pipeline.
#
#   PYTHONPATH=~/Documents/WSL/SANE uv run wse check-lens 'sane_wse:from_manifest(path=<this file>)' --zoo <indexed zoo>
#
# The head lives where it was dropped, in the wse repo, so this path is relative to a sibling checkout.
ae_trial_dir: sane_pretraining/{ds}/{TRIAL}
prediction_head: ../wse/SANE_test_models/{ds}/mlp/mlp_head_seed67.pt

# The head saw embeddings of tokens standardised per layer (DatasetTokens standardize=True). The
# statistics were never written out, so these are recomputed with the same formula over the head's
# own train split (seed-67 re-split of the zoo, {len(splits['train'])} models, epoch 8): per model, mean and
# std of each layer's [weight | bias] rows, then mean of the means and root-mean-square of the stds.
ignore_bn: true
""" + yaml.safe_dump({"standardization": stats}, sort_keys=False) + """
# Trained with map_to_canonical=True; alignment is off for the same reason as the ridge head
# (measured marginal, infeasible per step, and the cluster's reference model is not reproducible).
permutation_spec: null
clamp_min: null
""")

scratch = S / f"zoo_{ds}_mlp67_test400"
if not (scratch / "index.parquet").is_file():
    subprocess.run(["uv", "run", "python", "scripts/subset_zoo.py", "--zoo", str(ZOO), "--split-file", str(out / "splits_seed67.json"),
                    "--split", "test", "--n", "400", "--out", str(scratch)], cwd=WSE, check=True)
    subprocess.run(["uv", "run", "wse", "index", str(scratch), "--device", "cpu", "--download"], cwd=WSE, check=True)
spec = f"sane_wse:from_manifest(path={manifest})"
r = subprocess.run(["uv", "run", "wse", "check-lens", spec, "--zoo", str(scratch), "--device", "cpu"], cwd=WSE, capture_output=True, text=True)
print("check-lens:", [l for l in (r.stdout + r.stderr).splitlines() if l.startswith(("OK", "FAIL", "ERROR"))], flush=True)
assert r.returncode == 0, r.stderr[-2000:]

from wse.zoo import Zoo, batches, scan
from wse.lenses import load_lens
zoo = Zoo.open(scratch)
records = scan(scratch)
lens = load_lens(spec, zoo, device="cpu")
pos = {d: i for i, d in enumerate(dirs)}
mz, mp, cz, Y, ids = [], [], [], [], []
t0 = time.time()
for b in batches(records, zoo.arch, 64, device="cpu"):
    with torch.no_grad():
        mz.append(lens.embed(b.params.params)); mp.append(lens(b.params.params))
    idx = torch.tensor([pos[m] for m in b.model_ids]); cz.append(cache["z"][idx]); Y.append(cache["Y"][idx]); ids += list(b.model_ids)
sec = (time.time() - t0) / len(ids)
mz, mp, cz, Y = map(torch.cat, (mz, mp, cz, Y))
with torch.no_grad(): cp = lens.head(cz)
def r2(p, y):
    ok = ~torch.isnan(y).any(1); p, y = p[ok].double(), y[ok].double()
    r = 1 - ((y - p) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0); return float(r.mean()), float((y - p).abs().mean()), [round(v, 3) for v in r.tolist()]
cos = torch.nn.functional.cosine_similarity(mz, cz, dim=1)
idx = pd.read_parquet(scratch / "index.parquet"); idx = idx[idx.epoch == idx.epoch.max()].set_index("model_id").loc[ids]
ours = idx[[f"acc_class_{k}" for k in range(10)]].to_numpy(float)
rec = idx[[f"recorded_acc_class_{k}" for k in range(10)]].to_numpy(float)
ok = ~np.isnan(rec)
report = {
    "dataset": ds, "n_models": len(ids), "ms_per_model_cpu": round(sec * 1000, 1),
    "embedding_cosine_mean": round(float(cos.mean()), 4), "embedding_cosine_min": round(float(cos.min()), 4),
    "prediction_shift_vs_cached_mean": round(float((mp - cp).abs().mean()), 4), "prediction_shift_vs_cached_max": round(float((mp - cp).abs().max()), 4),
    "lens_vs_truth": dict(zip(["mean_r2", "mae", "per_class"], r2(mp, Y))), "head_on_cached_z_vs_truth": dict(zip(["mean_r2", "mae", "per_class"], r2(cp, Y))),
    "models_with_negative_prediction": int((mp < 0).any(1).sum()),
    "index_vs_cached_targets_max_abs": float(np.nanmax(np.abs(ours - Y.numpy().astype(float)))),
    "index_vs_zoo_recorded_max_abs": float(np.abs(ours - rec)[ok].max()), "index_vs_zoo_recorded_mean_abs": float(np.abs(ours - rec)[ok].mean()),
    "models_without_recorded": int((~ok).any(1).sum()),
}
(S / f"{ds}_v1_mlp_sanity.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1), flush=True)

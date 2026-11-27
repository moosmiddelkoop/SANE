"""Run one metanet experiment spec in its own single-threaded process.
usage: python worker.py <spec_name> <seed> <epochs> [n_train_models_frac]"""
import os, sys, json, time
os.environ["KMP_AFFINITY"] = "disabled"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("NT", "1")))
from toyzoo import K
from exp import *
from sklearn.metrics import r2_score

name, seed, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
frac = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
zoo = dict(np.load(os.environ["ZOO"]))
split = zoo["split"]; tr, va, te = split == "train", split == "val", split == "test"
if frac < 1:
    idx = np.nonzero(tr)[0]; rng = np.random.default_rng(123)
    keep = rng.choice(idx, int(round(frac * len(idx))), replace=False)
    tr = np.zeros_like(tr); tr[keep] = True

def stack(mask, ck=(0, 1, 2)):
    return (torch.tensor(np.concatenate([zoo["theta"][c][mask] for c in ck]), dtype=torch.float32),
            torch.tensor(np.concatenate([zoo["recall"][c][mask] for c in ck]), dtype=torch.float32))
Xtr, Ytr = stack(tr); Xva, Yva = stack(va); Xte, Yte = stack(te)

def std_stats(pre):
    Z = Xtr
    for f in pre: Z = f(Z)
    return Z.mean(0), Z.std(0) + 1e-6

PA = lambda x, g: permute_hidden(x, g)
RS = lambda x, g: random_rescale(x, g)
SN, CS = scale_normalize, canon_sort
CAN = lambda t: canon_sort(scale_normalize(t))
# per-tensor input scales for NeuronNet (from training data, after optional scale-norm)
def tensor_scales(pre):
    Z = Xtr[:20000]
    for f in pre: Z = f(Z)
    return [float(p.std()) for p in unflatten(Z)]

TASK = os.environ.get("TOYTASK", "blobs")
CEN = (-0.5, 0.5) if TASK == "blobs" else (0.0, 0.0)
GRID = (np.stack(np.meshgrid(np.linspace(-4, 3, 8), np.linspace(-3.5, 3.5, 8)), -1).reshape(-1, 2) if TASK == "blobs"
        else np.stack(np.meshgrid(np.linspace(-3.3, 3.3, 8), np.linspace(-3.3, 3.3, 8)), -1).reshape(-1, 2))
SPECS = {
    # ---- flat-MLP family
    "mlp_raw":            dict(mk=lambda: MLPMeta(422)),
    "mlp_aug":            dict(mk=lambda: MLPMeta(422), aug=[PA]),
    "mlp_aug_std":        dict(mk=lambda: MLPMeta(422, std=std_stats([])), aug=[PA]),
    "mlp_sn_aug_std":     dict(mk=lambda: MLPMeta(422, pre=[SN], std=std_stats([SN])), aug=[PA]),
    "mlp_canon_std":      dict(mk=lambda: MLPMeta(422, pre=[CS], std=std_stats([CS]))),
    "mlp_fullcanon_std":  dict(mk=lambda: MLPMeta(422, pre=[CAN], std=std_stats([CAN]))),
    "mlp_fullcanon_std_bce": dict(mk=lambda: MLPMeta(422, pre=[CAN], std=std_stats([CAN])), loss="bce"),
    "mlp_sn_aug_std_bce": dict(mk=lambda: MLPMeta(422, pre=[SN], std=std_stats([SN])), aug=[PA], loss="bce"),
    "mlp_aug_rs_std":     dict(mk=lambda: MLPMeta(422, std=std_stats([])), aug=[PA, RS]),
    # ---- permutation-invariant neuron-graph metanet
    "neuron_r1":          dict(mk=lambda: NeuronNet(64, 1, wscale=tensor_scales([]))),
    "neuron_r2":          dict(mk=lambda: NeuronNet(64, 2, wscale=tensor_scales([]))),
    "neuron_sn_r2":       dict(mk=lambda: NeuronNet(64, 2, pre=[SN], wscale=tensor_scales([SN]))),
    "neuron_sn_r2_bce":   dict(mk=lambda: NeuronNet(64, 2, pre=[SN], wscale=tensor_scales([SN])), loss="bce"),
    # ---- functional probing
    "probe_grid64":       dict(mk=lambda: ProbeNet(grid=GRID, learn=False)),
    "probe_rand16_fixed": dict(mk=lambda: ProbeNet(16, learn=False, center=CEN)),
    "probe_learn4":       dict(mk=lambda: ProbeNet(4, center=CEN)),
    "probe_learn8":       dict(mk=lambda: ProbeNet(8, center=CEN)),
    "probe_learn16":      dict(mk=lambda: ProbeNet(16, center=CEN)),
    "probe_learn32":      dict(mk=lambda: ProbeNet(32, center=CEN)),
    "probe_learn64":      dict(mk=lambda: ProbeNet(64, center=CEN)),
    "probe_learn16_bce":  dict(mk=lambda: ProbeNet(16, center=CEN), loss="bce"),
    "probe_learn64_bce":  dict(mk=lambda: ProbeNet(64, center=CEN), loss="bce"),
}
RUNS = os.environ.get("RUNS", "runs")
os.makedirs(RUNS, exist_ok=True)
sp = SPECS[name]
torch.manual_seed(seed)
net = sp["mk"]()
t0 = time.time()
net, best, hist = train(net, Xtr, Ytr, Xva, Yva, epochs=epochs, aug=sp.get("aug", ()), loss=sp.get("loss", "mse"),
                        seed=seed, lr=sp.get("lr", 1e-3), eval_every=5)
row = dict(name=name, seed=seed, epochs=epochs, frac=frac, n_train=len(Xtr), secs=time.time() - t0,
           params=sum(p.numel() for p in net.parameters()), hist=hist)
nf = len(Xva) // 3
for split_name, X, Y in [("val", Xva, Yva), ("test", Xte, Yte)]:
    P, r2, r2c = evaluate(net, X, Y)
    Yn = Y.numpy(); fin = slice(2 * nf, None)
    row[f"{split_name}_r2"] = float(r2)
    row[f"{split_name}_r2_final"] = float(r2_score(Yn[fin], P[fin]))
    row[f"{split_name}_mae"] = float(np.abs(P - Yn).mean())
    row[f"{split_name}_r2_c"] = [float(v) for v in r2c]
    row[f"{split_name}_r2_c_final"] = [float(v) for v in r2_score(Yn[fin], P[fin], multioutput="raw_values")]
    if split_name == "test":
        np.save(f"{RUNS}/pred_{name}_s{seed}_f{frac}.npy", P.astype(np.float32))
os.makedirs("runs", exist_ok=True)
torch.save(net.state_dict(), f"{RUNS}/{name}_s{seed}_f{frac}.pt")
json.dump(row, open(f"{RUNS}/{name}_s{seed}_f{frac}.json", "w"))
print(name, seed, frac, f"val {row['val_r2']:.4f} test {row['test_r2']:.4f} {row['secs']:.0f}s", flush=True)

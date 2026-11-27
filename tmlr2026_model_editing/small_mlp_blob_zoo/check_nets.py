import os, sys, numpy as np, torch; sys.path.insert(0, os.getcwd()); torch.set_num_threads(2)
import ublib as U
z = U.load_zoo(); te = np.where(z["split"] == "test")[0]
for (spec, s), path in U.NETS.items():
    if not os.path.exists(path): print(spec, s, 'missing'); continue
    net = U.load_net(spec, path, z)
    stored = np.load(f"{os.path.dirname(path)}/pred_{spec}_s{s}_f1.0.npy")[2 * len(te):]
    with torch.no_grad(): P = torch.cat([net(torch.tensor(z["theta"][-1, te[i:i+2000]])) for i in range(0, len(te), 2000)]).numpy()
    print(spec, s, 'max|diff| vs stored preds', float(np.abs(P - stored).max()))
M, C = U.edit_set(z); print('edit set', len(M), np.bincount(C), len(np.unique(M)))
R = z["recall"][-1]; ok = (R[z["split"]=="test"] > 0).all(1); print('no-dead test models', ok.sum(), [int(((R[z["split"]=="test"][:, c] >= 0.5) & ok).sum()) for c in range(6)])
print('rms theta median', float(np.sqrt((z["theta"][-1, M] ** 2).mean(1)).mean()), 'lr median', float(np.median(z["lr"][M])))

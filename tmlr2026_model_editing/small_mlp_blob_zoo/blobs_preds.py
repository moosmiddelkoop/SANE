"""Collect final-checkpoint test-split predictions (rows = np.where(split=='test'), ckpt 600) for every
blobs predictor into preds_final_test.npz."""
import os, sys, glob, numpy as np, torch
sys.path.insert(0, os.getcwd())
torch.set_num_threads(2)
from toyzoo import unflatten
from meta import MetaNet
Z = np.load('zoo_50k.npz'); split = Z['split']; TH = Z['theta']; REC = Z['recall']
te = np.where(split == 'test')[0]; tr = np.where(split == 'train')[0]; va = np.where(split == 'val')[0]
nte = len(te); out = {'Y': REC[-1, te], 'acc': Z['acc'][-1, te], 'model': te}
# worker.py predictions: test rows checkpoint-major, final block = last third
for f in sorted(glob.glob('rescored/runs/pred_*.npy') + glob.glob('runs_new/pred_*.npy')):
    P = np.load(f); assert len(P) == 3 * nte, f
    out[os.path.basename(f)[5:-4]] = P[2 * nte:]
    # sanity: worker order check against truth for the final block is implicit (same split mask)
# stored paper-recipe plain MLP (metanets_50k.pt['net_raw'], meta.MetaNet 5x256)
sd = torch.load('metanets_50k.pt', map_location='cpu', weights_only=False)
for k in ('net_raw', 'net_aug'):
    net = MetaNet(422, 6, 256, 5, 0.03); net.load_state_dict(sd[k]); net.eval()
    with torch.no_grad():
        out['legacy_' + k] = net(torch.tensor(TH[-1, te], dtype=torch.float32)).numpy()
# ridge on permutation-invariant weight statistics (toy analogue of meta_cnn.invariant_features)
def feats(theta):
    t = torch.as_tensor(theta, dtype=torch.float64); ps = unflatten(t); B = t.shape[0]; f = []
    for p in ps:
        q = p.reshape(B, -1)
        f += [q.pow(2).mean(1, keepdim=True).sqrt(), q.mean(1, keepdim=True), q.std(1, keepdim=True), q.abs().amax(1, keepdim=True)]
    W2, b2 = ps[4], ps[5]                       # (B,16,6), (B,6)
    f.append(torch.sort(W2.transpose(1, 2), dim=-1).values.reshape(B, -1))   # sorted incoming weights per class (6x16)
    f.append(b2); f.append(W2.norm(dim=1))
    x = torch.cat(f, -1); return torch.cat([x, torch.log1p(x.abs())], -1).numpy()
Ftr = np.concatenate([feats(TH[c, tr]) for c in range(3)]); Ytr = np.concatenate([REC[c, tr] for c in range(3)]).astype(np.float64)
Fva = feats(TH[-1, va]); Yva = REC[-1, va]; Fte = feats(TH[-1, te])
mu, sdv = Ftr.mean(0), Ftr.std(0) + 1e-8
std = lambda F: np.hstack([(F - mu) / sdv, np.ones((len(F), 1))])
X = std(Ftr); best = None
for a in [0.1, 1, 10, 100, 1000]:
    A = X.T @ X + a * np.eye(X.shape[1]); A[-1, -1] -= a
    W = np.linalg.solve(A, X.T @ Ytr); pv = np.clip(std(Fva) @ W, 0, 1)
    v = 1 - ((Yva - pv) ** 2).sum() / ((Yva - Yva.mean()) ** 2).sum()
    print('ridge alpha', a, 'val r2 final', round(v, 4))
    if a == 10: W10 = W
    if best is None or v > best[0]: best = (v, a, W)
out['ridge_wstats_a10'] = np.clip(std(Fte) @ W10, 0, 1)
out['ridge_wstats_valsel'] = np.clip(std(Fte) @ best[2], 0, 1); print('val-selected alpha', best[1])
out['zoo_mean_const'] = np.broadcast_to(REC[:, tr].reshape(-1, 6).mean(0), (nte, 6)).copy()
np.savez('preds_final_test.npz', **{k: np.asarray(v, np.float32) if k not in ('model',) else v for k, v in out.items()})
print(sorted(out.keys()), Ftr.shape)

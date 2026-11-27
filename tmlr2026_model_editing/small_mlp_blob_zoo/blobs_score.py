"""Headline scoring for the blobs predict panel. Interaction R2x is computed exactly as
sane_toy_pretrain.metrics()['r2_interaction']: final checkpoint, test split, acc>=0.5, truth and
prediction each double-centred by their own (model, class) means over the evaluated set,
R2x = 1 - SS(I(Y)-I(P)) / SS(I(Y) - mean) (mean of I(Y) is 0). Bootstrap: resample models with
replacement, re-estimate all means inside each resample."""
import numpy as np

def r2(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())

def dc(y):
    return y - y.mean(1, keepdims=True) - y.mean(0, keepdims=True) + y.mean()

def r2x(Y, P):
    return r2(dc(Y), dc(P))

def boot(Y, P, n=2000, seed=0):
    rng = np.random.default_rng(seed); M = len(Y)
    Y = np.asarray(Y, np.float64); P = np.asarray(P, np.float64)
    s = np.empty(n)
    for b in range(n):
        i = rng.integers(0, M, M); s[b] = r2x(Y[i], P[i])
    return np.percentile(s, [2.5, 97.5])

def score(Y, P, n_boot=2000, seed=0):
    lo, hi = boot(Y, P, n_boot, seed)
    return dict(r2x=r2x(Y, P), r2x_lo=float(lo), r2x_hi=float(hi), r2_raw=r2(Y, P),
                mse=float(((Y - P) ** 2).mean()), mae=float(np.abs(Y - P).mean()), n_test=int(len(Y)))

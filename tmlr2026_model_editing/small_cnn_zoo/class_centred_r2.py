"""Interaction R² (R²_int) for per-class recall, written out in plain NumPy.

R²_int is the fraction of the model × class interaction in test-set recall that a predictor explains.
The interaction is what remains of a (models × classes) table after removing each model's mean,
each class's mean, and adding back the grand mean (additive double centring, as in the residual of
a two-way ANOVA with one observation per cell). Truth and prediction are each double-centred with
their own means, so every additive prediction (per-model term + per-class term) scores exactly 0.

Inputs
------
Y : (M, K) true per-class recall of the M evaluated models (trained, acc >= 0.5, epoch 86, test split)
P : (M, K) predicted per-class recall for the same models, same row/column order
"""
import numpy as np


def double_centre(A):
    """A[m,k] - mean_k A[m,:] - mean_m A[:,k] + mean A.
    The row means and the column means each contain the grand mean, so subtracting both
    removes it twice; adding it back once leaves rows and columns that each average to 0."""
    A = np.asarray(A, dtype=float)
    return A - A.mean(axis=1, keepdims=True) - A.mean(axis=0, keepdims=True) + A.mean()


def interaction_r2(Y, P):
    Y = np.asarray(Y, dtype=float)
    P = np.asarray(P, dtype=float)
    assert Y.shape == P.shape and Y.ndim == 2

    Y_int = double_centre(Y)                   # true interaction
    P_int = double_centre(P)                   # predicted interaction (prediction's own means)

    sse_k = ((Y_int - P_int) ** 2).sum(axis=0) # (K,) error per class
    sst_k = (Y_int ** 2).sum(axis=0)           # (K,) interaction variance per class (mean is 0 by construction)

    r2_per_class = 1.0 - sse_k / sst_k         # supplement
    r2_pooled = 1.0 - sse_k.sum() / sst_k.sum()  # headline = np.average(r2_per_class, weights=sst_k)
    return r2_pooled, r2_per_class


def bootstrap_ci(Y, P, n_boot=1000, seed=0, level=95):
    """Percentile CI of the pooled R²_int over resampled models (rows); all means re-estimated per resample."""
    Y = np.asarray(Y, dtype=float); P = np.asarray(P, dtype=float)
    rng = np.random.default_rng(seed)
    M = len(Y)
    stats = [interaction_r2(Y[i], P[i])[0] for i in (rng.integers(0, M, M) for _ in range(n_boot))]
    a = (100 - level) / 2
    return np.percentile(stats, [a, 100 - a])

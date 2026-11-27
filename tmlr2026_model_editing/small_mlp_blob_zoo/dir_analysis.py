"""Analysis of per-class edit directions (outputs of dir_walk.py)."""
import numpy as np
K, H = 6, 16
SL = {"W0": slice(0, 32), "b0": slice(32, 48), "W1": slice(48, 304), "b1": slice(304, 320),
      "W2": slice(320, 416), "b2": slice(416, 422)}
HID = np.r_[0:320]; OUT = np.r_[320:422]
SCOPES = {"all": np.r_[0:422], "hidden": HID, "output": OUT}

def permute(d, P):
    """d (..., 422) per model, P (2,16) SANE permutation -> aligned (same gather semantics as sane_psi.permute)."""
    p0, p1 = P
    W0 = d[..., SL["W0"]].reshape(*d.shape[:-1], 2, H)[..., :, p0]
    b0 = d[..., SL["b0"]][..., p0]
    W1 = d[..., SL["W1"]].reshape(*d.shape[:-1], H, H)[..., p0, :][..., :, p1]
    b1 = d[..., SL["b1"]][..., p1]
    W2 = d[..., SL["W2"]].reshape(*d.shape[:-1], H, K)[..., p1, :]
    b2 = d[..., SL["b2"]]
    sh = d.shape[:-1]
    return np.concatenate([W0.reshape(*sh, -1), b0, W1.reshape(*sh, -1), b1, W2.reshape(*sh, -1), b2], -1)

def unit(x, eps=1e-20):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)

def within_model(D):
    """D (NM,K,N) -> per-model KxK cosine (NM,K,K)."""
    U = unit(D); return np.einsum("mcn,mkn->mck", U, U)

def cross(D, G):
    """cos(D[m,c], G[m,k]) -> (NM,K,K)."""
    return np.einsum("mcn,mkn->mck", unit(D), unit(G))

def offdiag(S):
    m = ~np.eye(S.shape[-1], dtype=bool); return S[..., m]

def selectivity(A):
    """A (NM,K,K) cos(edit_c, ref_k): diag mean, offdiag mean, index (diag - offdiag mean per row), top-1 accuracy."""
    NM = A.shape[0]; d = np.einsum("mcc->mc", A)
    od = (A.sum(-1) - d) / (K - 1)
    top1 = (A.argmax(-1) == np.arange(K)[None]).mean()
    return dict(diag=d.mean(), off=od.mean(), index=(d - od).mean(), index_rows=(d - od).ravel(), top1=top1)

def centroid_decode(Da, cls_cv=True):
    """Cross-model class decoding from aligned unit edit directions Da (NM,K,N):
    leave-one-model-out nearest-class-centroid (cosine). Returns accuracy and the KxK mean-cosine matrix
    between models (same-class vs different-class, excluding same-model pairs)."""
    U = unit(Da); NM = U.shape[0]
    S = U.sum(0)                                  # (K,N) class sums
    acc = []
    for m in range(NM):
        cen = unit(S - U[m]); sc = U[m] @ cen.T    # (K,K): row = true class, col = centroid class
        acc.append((sc.argmax(1) == np.arange(K)))
    # between-model mean cosine matrix: mean_{m!=m'} <U[m,c], U[m',k]> = (|S_c . S_k| - sum_m U[m,c].U[m,k]) / (NM(NM-1))
    G = S @ S.T - np.einsum("mcn,mkn->ck", U, U)
    return np.mean(acc), G / (NM * (NM - 1))

def rv_like(D, rng):
    """Random vectors norm-matched per layer (W,b) block to D, as ublib.random_matched."""
    out = np.zeros_like(D)
    for w, b in [("W0", "b0"), ("W1", "b1"), ("W2", "b2")]:
        idx = np.r_[SL[w], SL[b]]
        tau = np.linalg.norm(D[..., idx], axis=-1, keepdims=True)
        v = rng.standard_normal(D[..., idx].shape).astype(np.float32)
        out[..., idx] = unit(v) * tau
    return out

def bias_like(D):
    """Output-bias shift on b_c only (Bias control), norm-matched."""
    out = np.zeros_like(D); NM = D.shape[0]; n = np.linalg.norm(D, axis=-1)
    for c in range(K): out[:, c, 416 + c] = -n[:, c]
    return out

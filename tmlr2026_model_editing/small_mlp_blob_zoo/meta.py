"""Metanetwork psi: flat weights -> per-class recall (paper recipe, scaled down)."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


class MetaNet(nn.Module):
    def __init__(self, n_in, n_out, hidden=256, n_hidden=4, dropout=0.03):
        super().__init__()
        layers, d = [], n_in
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, n_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, theta):
        return torch.sigmoid(self.net(theta))


def load_zoo(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def stack_ckpts(zoo, mask, ckpts=(0, 1, 2)):
    """Select models by boolean mask over models, stacking chosen checkpoints."""
    th = np.concatenate([zoo["theta"][c][mask] for c in ckpts])
    rec = np.concatenate([zoo["recall"][c][mask] for c in ckpts])
    return th.astype(np.float32), rec.astype(np.float32)


def metrics(y_true, y_pred):
    return dict(r2=r2_score(y_true, y_pred), mse=mean_squared_error(y_true, y_pred),
                mae=mean_absolute_error(y_true, y_pred),
                r2_per_class=r2_score(y_true, y_pred, multioutput="raw_values"))


def permute_hidden(theta, gen=None):
    """Apply an independent random permutation of the hidden units of each hidden
    layer to every model in theta (B, N). Function-preserving symmetry of the MLP."""
    from toyzoo import unflatten, flatten, LAYERS
    ps = unflatten(theta)
    B = theta.shape[0]
    n_layers = len(LAYERS)
    ps = list(ps)
    for l in range(n_layers - 1):               # hidden layers only
        h = LAYERS[l][1]
        P = torch.argsort(torch.rand(B, h, generator=gen), dim=1)     # (B,h)
        W, b = ps[2 * l], ps[2 * l + 1]
        ps[2 * l] = torch.gather(W, 2, P[:, None, :].expand(-1, W.shape[1], -1))
        ps[2 * l + 1] = torch.gather(b, 1, P)
        Wn = ps[2 * (l + 1)]
        ps[2 * (l + 1)] = torch.gather(Wn, 1, P[:, :, None].expand(-1, -1, Wn.shape[2]))
    return flatten(ps)


def train_metanet(Xtr, Ytr, Xva, Yva, epochs=300, lr=1e-3, wd=1e-5, batch=256, seed=0,
                  hidden=256, n_hidden=4, dropout=0.03, augment=False, verbose=False):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    Xva_t, Yva_t = torch.tensor(Xva), torch.tensor(Yva)
    net = MetaNet(Xtr.shape[1], Ytr.shape[1], hidden, n_hidden, dropout)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = -1e9, None
    n = len(Xtr_t)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = permute_hidden(Xtr_t[idx], gen) if augment else Xtr_t[idx]
            loss = ((net(xb) - Ytr_t[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                r2 = r2_score(Yva, net(Xva_t).numpy())
            if r2 > best:
                best, best_state = r2, {k: v.clone() for k, v in net.state_dict().items()}
            if verbose:
                print(f"ep {ep+1} val R2 {r2:.4f}")
    net.load_state_dict(best_state); net.eval()
    return net, best

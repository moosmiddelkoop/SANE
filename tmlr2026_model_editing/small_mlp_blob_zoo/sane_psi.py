"""SANE-W psi for the blobs zoo: theta (raw, unaligned) -> per-model git-re-basin permutation (from sane.pt)
-> tokenise -> standardise -> frozen SANE encoder -> frozen token head -> per-class recall (B,K).
Re-implements tokenize/permute/standardise exactly as in sane_toy_pretrain_blobs.py; differentiable in theta."""
import torch, torch.nn as nn, torch.nn.functional as F
import toyzoo as tz
from sane_vendor.def_AE import AE
H, K = tz.H, tz.K; D_T, N_TOK = H + 1, 2 * H + K
LAYER_FAN = [(tz.D_IN, H), (H, H), (H, K)]; OUT_TOK = slice(2 * H, N_TOK)
MASK = torch.zeros(N_TOK, D_T, dtype=torch.bool); POS = torch.zeros(N_TOK, 3, dtype=torch.long); LID = torch.zeros(N_TOK, dtype=torch.long)
n = 0
for l, (i, o) in enumerate(LAYER_FAN):
    for r in range(o):
        MASK[n, : i + 1] = True; POS[n] = torch.tensor([n, l, r]); LID[n] = l; n += 1

def tokenize(theta):
    ps = tz.unflatten(theta); out = []
    for l, (i, o) in enumerate(LAYER_FAN):
        t = torch.cat([ps[2 * l].transpose(1, 2), ps[2 * l + 1][:, :, None]], 2)
        out.append(F.pad(t, (0, D_T - i - 1)))
    return torch.cat(out, 1)

def permute(theta, P):
    W0, b0, W1, b1, W2, b2 = tz.unflatten(theta); B = theta.shape[0]; p0, p1 = P[:, 0], P[:, 1]
    W0 = W0.gather(2, p0[:, None, :].expand(B, W0.shape[1], H)); b0 = b0.gather(1, p0)
    W1 = W1.gather(1, p0[:, :, None].expand(B, H, H)).gather(2, p1[:, None, :].expand(B, H, H)); b1 = b1.gather(1, p1)
    W2 = W2.gather(1, p1[:, :, None].expand(B, H, K))
    return tz.flatten([W0, b0, W1, b1, W2, b2])

class TokenHead(nn.Module):
    def __init__(s, lat, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * lat + K, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
    def forward(s, z):
        B = z.shape[0]
        x = torch.cat([z.mean(1, keepdim=True).expand(B, K, z.shape[2]), z[:, OUT_TOK], torch.eye(K)[None].expand(B, K, K)], 2)
        return torch.sigmoid(s.net(x)[..., 0])

class SanePsi:
    """Callable psi(theta) for a fixed list of zoo model ids (rows of theta correspond to `models`)."""
    def __init__(self, path, models):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.ae = AE(d["cfg"]); self.ae.load_state_dict(d["ae"]); self.ae.eval()
        lat = d["cfg"]["ae:lat_dim"]; self.head = TokenHead(lat); self.head.load_state_dict(d["heads"]["sane_token"]); self.head.eval()
        for p in list(self.ae.parameters()) + list(self.head.parameters()): p.requires_grad_(False)
        self.P = d["perm"][torch.as_tensor(models)]
        self.mu = d["mu"][LID].view(1, N_TOK, 1); self.sg = d["sigma"][LID].view(1, N_TOK, 1)
    def __call__(self, theta, P=None):
        P = self.P if P is None else P
        t = (tokenize(permute(theta, P)) - self.mu) / self.sg * MASK
        z = self.ae.forward_encoder(t, POS[None].expand(len(t), N_TOK, 3))
        return self.head(z)

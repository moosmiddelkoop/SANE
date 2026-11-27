"""Windowed SANE autoencoder over ResNet-18 row tokens (vendored SANE AE, MIT) + whole-model encoding/decoding helpers.
No permutation alignment (deviation from SANE: no git-re-basin for residual nets); augmentation = multiplicative noise."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import resnet_zoo as rz
from sane_vendor.def_AE import AE


def ae_config(d_model=1024, nhead=8, num_layers=4, lat_dim=128, window=512, dropout=0.0):
    return {"ae:i_dim": rz.D_T, "ae:d_model": d_model, "ae:nhead": nhead, "ae:num_layers": num_layers,
            "ae:lat_dim": lat_dim, "training::windowsize": window, "ae:dropout": dropout,
            "ae:max_positions": [rz.N_TOK, rz.N_LAYERS, 512], "ae:transformer_type": "gpt2"}


class SaneResNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg; self.ae = AE(cfg); self.W = cfg["training::windowsize"]; self.lat = cfg["ae:lat_dim"]
        self.register_buffer("pos", rz.POS.clone()); self.register_buffer("mask", rz.MASK.clone())

    def forward(self, tok, pos):           # training windows: (B, W, D_T), (B, W, 3) -> z, y, zp
        return self.ae(tok, pos)

    def encode_windows(self, tok, pos):
        return self.ae.forward_encoder(tok, pos)

    def decode_windows(self, z, pos):
        return self.ae.forward_decoder(z, pos)

    def window_starts(self):
        return list(range(0, rz.N_TOK, self.W))

    @torch.no_grad()
    def encode_model(self, tok_std, chunk=32):
        """tok_std (B, N_TOK, D_T) standardised tokens of B models -> z (B, N_TOK, lat), windows at fixed offsets."""
        B = tok_std.shape[0]; out = torch.empty(B, rz.N_TOK, self.lat, device=tok_std.device, dtype=torch.float32)
        for b in range(B):
            starts = self.window_starts()
            for i in range(0, len(starts), chunk):
                ss = starts[i:i + chunk]
                xs = [tok_std[b, s:s + self.W] for s in ss]; ps = [self.pos[s:s + self.W] for s in ss]
                if len(xs) > 1 and xs[-1].shape[0] != xs[0].shape[0]:   # ragged last window
                    out[b, ss[-1]:ss[-1] + xs[-1].shape[0]] = self.encode_windows(xs[-1][None].float(), ps[-1][None])[0].float()
                    xs, ps, ss = xs[:-1], ps[:-1], ss[:-1]
                z = self.encode_windows(torch.stack(xs).float(), torch.stack(ps))
                for j, s in enumerate(ss):
                    out[b, s:s + self.W] = z[j].float()
        return out

    @torch.no_grad()
    def decode_model(self, z, chunk=32):
        """z (B, N_TOK, lat) -> standardised tokens (B, N_TOK, D_T) (padding entries masked to 0)."""
        B = z.shape[0]; out = torch.empty(B, rz.N_TOK, rz.D_T, device=z.device, dtype=torch.float32)
        for b in range(B):
            starts = self.window_starts()
            for i in range(0, len(starts), chunk):
                ss = starts[i:i + chunk]
                zs = [z[b, s:s + self.W] for s in ss]; ps = [self.pos[s:s + self.W] for s in ss]
                if len(zs) > 1 and zs[-1].shape[0] != zs[0].shape[0]:
                    out[b, ss[-1]:ss[-1] + zs[-1].shape[0]] = self.decode_windows(zs[-1][None], ps[-1][None])[0].float()
                    zs, ps, ss = zs[:-1], ps[:-1], ss[:-1]
                y = self.decode_windows(torch.stack(zs), torch.stack(ps))
                for j, s in enumerate(ss):
                    out[b, s:s + self.W] = y[j].float()
        return out * self.mask

    def encode_slice(self, tok_std_full, sl):
        """Differentiable encoding of the window(s) covering token slice sl (e.g. FC_TOKENS) of a batch of models.
        tok_std_full: (B, N_TOK, D_T). Returns z of the covering windows (B, L, lat) and the covering slice."""
        s0 = (sl.start // self.W) * self.W; s1 = min(rz.N_TOK, ((sl.stop - 1) // self.W + 1) * self.W)
        cov = slice(s0, s1)
        z = self.encode_windows(tok_std_full[:, cov].float(), self.pos[cov][None].expand(tok_std_full.shape[0], -1, -1))
        return z, cov

    def decode_slice(self, z_cov, cov):
        y = self.decode_windows(z_cov, self.pos[cov][None].expand(z_cov.shape[0], -1, -1))
        return y * self.mask[cov]


def save_sane(path, sane, heads, meta):
    torch.save({"cfg": sane.cfg, "ae": sane.ae.state_dict(),
                "heads": {k: (type(v).__name__, getattr(v, "kind", None), v.state_dict()) for k, v in heads.items()}, "meta": meta}, path)


def load_sane(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    sane = SaneResNet(ck["cfg"]).to(device); sane.ae.load_state_dict(ck["ae"]); sane.eval()
    heads = {}
    for k, (cls, kind, sd) in ck["heads"].items():
        h = TokenHead(ck["cfg"]["ae:lat_dim"]) if cls == "TokenHead" else ZBarHead(ck["cfg"]["ae:lat_dim"], kind=kind)
        h.load_state_dict(sd); heads[k] = h.to(device).eval()
    return sane, heads, ck


# ---------------------------------------------------------------- property heads (features computed by the caller)
class ZBarHead(nn.Module):
    """Head on z̄ (mean token latent). 'linear' (ridge / linear probe) or 'mlp'. Output: 100 per-class recalls."""

    def __init__(self, lat, kind="linear", hidden=512):
        super().__init__(); self.kind = kind
        self.net = nn.Linear(lat, rz.NC) if kind == "linear" else nn.Sequential(
            nn.Linear(lat, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, rz.NC))

    def forward(self, zbar, zfc=None):
        out = self.net(zbar)
        return out if self.kind == "linear" else torch.sigmoid(out)


class TokenHead(nn.Module):
    """Per-class head: recall_c = f([z̄, z_fc(c) tokens, onehot(c)]), f shared across classes."""

    def __init__(self, lat, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(lat * (1 + rz.TOK_PER_CLASS) + rz.NC, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, zbar, zfc):
        """zbar (B, lat); zfc (B, 100, TOK_PER_CLASS, lat) -> (B, 100)"""
        B = zbar.shape[0]
        x = torch.cat([zbar[:, None].expand(B, rz.NC, -1), zfc.reshape(B, rz.NC, -1),
                       torch.eye(rz.NC, device=zbar.device)[None].expand(B, -1, -1)], 2)
        return torch.sigmoid(self.net(x)[..., 0])


def zfc_from_z(z):
    """z (B, N_TOK, lat) -> fc-token latents (B, 100, TOK_PER_CLASS, lat)."""
    return z[:, rz.FC_TOKENS].reshape(z.shape[0], rz.NC, rz.TOK_PER_CLASS, -1)

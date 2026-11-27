"""psi(theta) = fc-only flat MLP (train_mlpfc_ep20.py checkpoint) as a differentiable map flat theta (B,P) -> recall (B,100).
Depends only on the 51,300 fc parameters, so its weight-space gradient is zero outside the fc layer (gradient_scope 'output')."""
import torch, torch.nn as nn

class FcMlpPsi(nn.Module):
    def __init__(self, ckpt_path, hidden=256, n_hidden=5, dropout=0.03, K=100):
        super().__init__()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False); D = ck["D"]; self.lo, self.hi = ck["fc_lo"], ck["fc_hi"]
        self.register_buffer("mu", torch.zeros(D)); self.register_buffer("sd", torch.ones(D))
        layers, d = [], D
        for _ in range(n_hidden): layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout)]; d = hidden
        layers += [nn.Linear(d, K)]; self.net = nn.Sequential(*layers)
        self.load_state_dict(ck["state_dict"])
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()
    def set_batch(self, *args): pass          # no per-model state (BN buffers / latents not used)
    def forward(self, theta): return torch.sigmoid(self.net((theta[:, self.lo:self.hi] - self.mu) / self.sd))

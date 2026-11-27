"""psi(theta) for editing: the trained ResNet-18 channel-graph metanet as a differentiable map flat theta (B,P) -> recall (B,100).
BN running statistics are not part of theta; they are fixed per model and set per batch with set_batch(bufs)."""
import json, os
import numpy as np, torch, torch.nn as nn
import resnet_zoo as rz
from resnet_gmn import ResNetGraphNet, normalise


class GmnPsi(nn.Module):
    def __init__(self, ckpt_pt, norm_stats_json, d=64):
        super().__init__()
        self.net = ResNetGraphNet(64, rz.NC, d, fused=True, resid=True)
        self.net.load_state_dict(torch.load(ckpt_pt, map_location="cpu", weights_only=False))
        self.st = {k: tuple(v) for k, v in json.load(open(norm_stats_json)).items()}
        for p in self.parameters(): p.requires_grad_(False)
        self.eval(); self.bufs = None

    def set_batch(self, bufs):
        """bufs (B, NB) BN running stats of the models about to be edited (same order as theta rows)."""
        self.bufs = bufs

    def forward(self, theta):
        B = theta.shape[0]
        p = rz.flat_to_params(theta); p.update(rz.buffers_to_dict(self.bufs[:B]))
        return self.net(normalise(p, self.st))


def load_gmn_psi(runs_root, seed, d=64, epochs=300, dev="cuda"):
    """runs_root/s{seed}/gmn_d{d}_s{seed}_e{epochs}_{best_val_r2x.pt, norm_stats.json}"""
    tag = f"gmn_d{d}_s{seed}_e{epochs}"
    base = os.path.join(runs_root, f"s{seed}", tag)
    return GmnPsi(base + "_best_val_r2x.pt", base + "_norm_stats.json", d=d).to(dev)


if __name__ == "__main__":   # CPU self-test with synthetic weights: gradient flows, permutation invariance, batch consistency
    import resnet_gmn as rg, tempfile
    torch.manual_seed(0); g = torch.Generator().manual_seed(0)
    net = ResNetGraphNet(64, rz.NC, 64, fused=True, resid=True)
    p = rg.random_theta(2, gen=g)
    st = rg.norm_stats(p)
    with tempfile.TemporaryDirectory() as td:
        torch.save(net.state_dict(), td + "/n.pt"); json.dump({k: list(v) for k, v in st.items()}, open(td + "/s.json", "w"))
        psi = GmnPsi(td + "/n.pt", td + "/s.json")
    # flatten synthetic params into theta / bufs in state-dict order
    th = torch.cat([p[k].reshape(2, -1) for k, _ in rz.PARAM_SPECS], 1); bf = torch.cat([p[k].reshape(2, -1) for k, _ in rz.BUFFER_KEYS], 1)
    assert th.shape[1] == rz.P and bf.shape[1] == rz.NB
    psi.set_batch(bf)
    th_ = th.clone().requires_grad_(True); out = psi(th_); out[:, 3].sum().backward()
    print("out", out.shape, out.min().item(), out.max().item(), "grad norm", th_.grad.norm().item(), "nonzero frac", (th_.grad != 0).float().mean().item())
    # permutation invariance: permute the synthetic model (function-preserving) and compare
    q = rg.permute_resnet(p, g)
    thq = torch.cat([q[k].reshape(2, -1) for k, _ in rz.PARAM_SPECS], 1); bq = torch.cat([q[k].reshape(2, -1) for k, _ in rz.BUFFER_KEYS], 1)
    psi.set_batch(bq); outq = psi(thq)
    print("perm invariance max |diff|", (out.detach() - outq).abs().max().item())
    # batch consistency: single row equals row in batch
    psi.set_batch(bf[:1]); o1 = psi(th[:1]); print("batch consistency", (o1 - out[:1].detach()).abs().max().item())
    # gradient restricted to fc indices is nonzero
    fc = rz.param_indices_of_tokens(rz.FC_TOKENS); print("fc idx", len(fc), "fc grad norm", th_.grad[:, fc].norm().item(), "conv grad norm", th_.grad.norm().item())

"""Channel-graph metanet for CIFAR ResNet-18 (generalises meta_cnn.ChannelGraphNet).
Nodes = output channels of every conv (BN params as node features), 3 input nodes, 100 class nodes.
Edges = conv kernels (in, out, k*k) ; downsample 1x1 edges ; identity skip edges (diagonal) ; fc edges.
One forward message pass in topological order, separate weights per step, mean aggregation,
message relu(A h_src + B e) -> MLP, as in ChannelGraphNet (rounds=1).
fused=True moves the last Linear of the message MLP after the mean (mathematically identical)."""
import math, time, json, sys, os
import torch, torch.nn as nn, torch.nn.functional as F

def mlp(d_in, d, d_out, n=2):
    layers, cur = [], d_in
    for _ in range(n):
        layers += [nn.Linear(cur, d), nn.ReLU()]; cur = d
    layers += [nn.Linear(cur, d_out)]
    return nn.Sequential(*layers)

def resnet18_spec(width=64, n_cls=100):
    """List of (name, shape) for a CIFAR ResNet-18 (3x3 conv1, no maxpool), torchvision naming."""
    W = [width, 2*width, 4*width, 8*width]
    spec = [("conv1.weight", (W[0], 3, 3, 3))] + [(f"bn1.{p}", (W[0],)) for p in ("weight","bias","running_mean","running_var")]
    cin = W[0]
    for li, c in enumerate(W, 1):
        for bi in range(2):
            pre = f"layer{li}.{bi}"
            s_in = cin if bi == 0 else c
            spec.append((f"{pre}.conv1.weight", (c, s_in, 3, 3)))
            spec += [(f"{pre}.bn1.{p}", (c,)) for p in ("weight","bias","running_mean","running_var")]
            spec.append((f"{pre}.conv2.weight", (c, c, 3, 3)))
            spec += [(f"{pre}.bn2.{p}", (c,)) for p in ("weight","bias","running_mean","running_var")]
            if bi == 0 and s_in != c:
                spec.append((f"{pre}.downsample.0.weight", (c, s_in, 1, 1)))
                spec += [(f"{pre}.downsample.1.{p}", (c,)) for p in ("weight","bias","running_mean","running_var")]
        cin = c
    spec += [("fc.weight", (n_cls, W[3])), ("fc.bias", (n_cls,))]
    return spec

def bn_feats(p, pre):
    """inputs are pre-normalised (see normalise): running_var already log-transformed"""
    return torch.stack([p[f"{pre}.{k}"] for k in ("weight", "bias", "running_mean", "running_var")], -1)   # (B,C,4)


def norm_stats(p):
    """per-tensor (mu, sigma) from a dict of (B, ...) tensors of training models; running_var in log space"""
    st = {}
    for k, v in p.items():
        x = torch.log(v + 1e-5) if k.endswith("running_var") else v
        x = x.double()
        st[k] = (x.mean().item(), max(x.std().item(), 1e-8))
    return st


def normalise(p, st):
    """one scale per tensor, then asinh (as meta_cnn.ChannelGraphNet); exact per-tensor affine map, so permutation-equivariant"""
    out = {}
    for k, v in p.items():
        x = torch.log(v + 1e-5) if k.endswith("running_var") else v
        mu, sg = st[k]
        out[k] = torch.asinh((x - mu) / sg)
    return out

class ResNetGraphNet(nn.Module):
    def __init__(self, width=64, n_cls=100, d=64, fused=False, resid=True):
        super().__init__()
        self.width, self.n_cls, self.d, self.fused, self.resid = width, n_cls, d, fused, resid
        self.spec = resnet18_spec(width, n_cls)
        self.blocks = [(li, bi) for li in range(1, 5) for bi in range(2)]
        self.inp_emb = nn.Parameter(torch.randn(3, d) * 0.1)
        def edge_mod(f):  # message module for one edge type/step
            return nn.ModuleDict(dict(h=nn.Linear(d, d), w=nn.Linear(f, d, bias=False), o=mlp(d, d, d, 1)))
        self.m = nn.ModuleDict(); self.enc = nn.ModuleDict(); self.upd = nn.ModuleDict()
        self.m["stem"] = edge_mod(9); self.enc["stem"] = mlp(4, d, d); self.upd["stem"] = mlp(2*d, d, d)
        for li, bi in self.blocks:
            k = f"l{li}b{bi}"
            self.m[k+"c1"] = edge_mod(9); self.enc[k+"c1"] = mlp(4, d, d); self.upd[k+"c1"] = mlp(2*d, d, d)
            self.m[k+"c2"] = edge_mod(9)
            ds = (bi == 0 and li > 1)
            if ds: self.m[k+"ds"] = edge_mod(1)
            else:  self.m[k+"id"] = nn.Linear(d, d)               # identity skip: one diagonal edge per channel
            self.enc[k+"c2"] = mlp(8 if ds else 4, d, d); self.upd[k+"c2"] = mlp(3*d, d, d)
        self.m["fc"] = edge_mod(1)
        self.ln = nn.ModuleDict({k: nn.LayerNorm(d) for k in self.upd.keys()})
        self.cls_emb = nn.Parameter(torch.randn(n_cls, d) * 0.1)
        self.out = mlp(3*d + 1, 2*d, 1, 2)

    def _msg(self, hs, E, mod):
        # hs (B,S,d), E (B,S,T,f) -> (B,T,d), mean over sources S
        z = torch.relu(mod["h"](hs)[:, :, None, :] + mod["w"](E))
        o = mod["o"]
        if self.fused:   # Linear2(mean(relu(Linear1 z))) == mean(Linear2(relu(Linear1 z)))
            return o[2](torch.relu(o[0](z)).mean(1))
        return o(z).mean(1)

    @staticmethod
    def _edges(w):   # conv weight (B,out,in,kh,kw) -> (B,in,out,kh*kw)
        B, co, ci = w.shape[:3]
        return w.permute(0, 2, 1, 3, 4).reshape(B, ci, co, -1)

    def logits(self, p):
        B = p["fc.weight"].shape[0]
        h_in = self.inp_emb[None].expand(B, -1, -1)
        h = self.upd["stem"](torch.cat([self.enc["stem"](bn_feats(p, "bn1")),
                                        self._msg(h_in, self._edges(p["conv1.weight"]), self.m["stem"])], -1))
        for li, bi in self.blocks:
            k, pre = f"l{li}b{bi}", f"layer{li}.{bi}"
            m1 = self._msg(h, self._edges(p[pre+".conv1.weight"]), self.m[k+"c1"])
            h1 = self.upd[k+"c1"](torch.cat([self.enc[k+"c1"](bn_feats(p, pre+".bn1")), m1], -1))
            if self.resid: h1 = self.ln[k+"c1"](h1 + m1)
            m2 = self._msg(h1, self._edges(p[pre+".conv2.weight"]), self.m[k+"c2"])
            if (k+"ds") in self.m:
                ms = self._msg(h, self._edges(p[pre+".downsample.0.weight"]), self.m[k+"ds"])
                nf = torch.cat([bn_feats(p, pre+".bn2"), bn_feats(p, pre+".downsample.1")], -1)
            else:
                ms = self.m[k+"id"](h); nf = bn_feats(p, pre+".bn2")
            hn = self.upd[k+"c2"](torch.cat([self.enc[k+"c2"](nf), m2, ms], -1))
            h = self.ln[k+"c2"](hn + ms + m2) if self.resid else hn
        Eo = p["fc.weight"].transpose(1, 2)[..., None]               # (B,512,100,1)
        r = self._msg(h, Eo, self.m["fc"])                             # (B,100,d)
        g = h.mean(1, keepdim=True).expand(-1, self.n_cls, -1)
        e = self.cls_emb[None].expand(B, -1, -1)
        return self.out(torch.cat([r, g, e, p["fc.bias"][..., None]], -1))[..., 0]

    def forward(self, p):
        return torch.sigmoid(self.logits(p))

def graph_stats(width=64, n_cls=100):
    spec = dict(resnet18_spec(width, n_cls))
    n_params = sum(math.prod(s) for k, s in spec.items() if "running" not in k)
    n_buf = sum(math.prod(s) for k, s in spec.items() if "running" in k)
    e9 = sum(s[0]*s[1] for k, s in spec.items() if k.endswith("weight") and len(s) == 4 and s[2] == 3)
    e1 = sum(s[0]*s[1] for k, s in spec.items() if "downsample.0" in k)
    efc = math.prod(spec["fc.weight"])
    nodes = 3 + sum(s[0] for k, s in spec.items() if k.endswith("conv1.weight") or k.endswith("conv2.weight")) + n_cls
    eid = sum(spec[f"layer{li}.{bi}.conv2.weight"][0] for li in range(1,5) for bi in range(2) if not (bi == 0 and li > 1))
    return dict(n_params=n_params, n_bn_buffers=n_buf, nodes=nodes, edges_3x3=e9, edges_1x1_downsample=e1,
                edges_fc=efc, edges_identity=eid, edges_total=e9+e1+efc+eid)

def random_theta(B, width=64, n_cls=100, device="cpu", dtype=torch.float32, gen=None):
    p = {}
    for k, s in resnet18_spec(width, n_cls):
        if "running_var" in k: v = torch.rand((B,)+s, generator=gen) + 0.5
        elif k.endswith("bn1.weight") or k.endswith("bn2.weight") or k.endswith("downsample.1.weight"): v = 1 + 0.1*torch.randn((B,)+s, generator=gen)
        elif len(s) >= 2: v = torch.randn((B,)+s, generator=gen) * math.sqrt(2.0 / math.prod(s[1:]))
        else: v = 0.1 * torch.randn((B,)+s, generator=gen)
        p[k] = v.to(device=device, dtype=dtype)
    return p

def permute_resnet(p, gen):
    """Function-preserving channel permutation: one perm per stage residual stream + one per block-internal conv1 output."""
    q = {k: v.clone() for k, v in p.items()}
    W = [p["conv1.weight"].shape[1]] + [p[f"layer{li}.0.conv2.weight"].shape[1] for li in range(1,5)]
    def perm_out(prefix_w, bn, P):
        q[prefix_w] = q[prefix_w][:, P]
        for s in ("weight","bias","running_mean","running_var"): q[f"{bn}.{s}"] = q[f"{bn}.{s}"][:, P]
    def perm_in(key, P): q[key] = q[key][:, :, P]
    # stage streams: stem output = layer1 stream
    prev = None
    for li in range(1, 5):
        c = p[f"layer{li}.0.conv2.weight"].shape[1]
        P = torch.randperm(c, generator=gen)
        if li == 1:
            perm_out("conv1.weight", "bn1", P)
        for bi in range(2):
            pre = f"layer{li}.{bi}"
            # block input
            Pin = P if not (bi == 0 and li > 1) else prev
            perm_in(pre+".conv1.weight", Pin)
            if (pre+".downsample.0.weight") in q:
                perm_in(pre+".downsample.0.weight", prev); perm_out(pre+".downsample.0.weight", pre+".downsample.1", P)
            Q = torch.randperm(c, generator=gen)
            perm_out(pre+".conv1.weight", pre+".bn1", Q); perm_in(pre+".conv2.weight", Q)
            perm_out(pre+".conv2.weight", pre+".bn2", P)
        prev = P
    q["fc.weight"] = q["fc.weight"][:, :, prev]
    return q

# ---------------- minimal CIFAR ResNet-18 (functional, from a state dict) for function checks / eval timing
def resnet_forward(sd, x):
    def bn(t, pre): return F.batch_norm(t, sd[pre+".running_mean"], sd[pre+".running_var"], sd[pre+".weight"], sd[pre+".bias"], False, 0.0, 1e-5)
    h = F.relu(bn(F.conv2d(x, sd["conv1.weight"], padding=1), "bn1"))
    for li in range(1, 5):
        for bi in range(2):
            pre = f"layer{li}.{bi}"; stride = 2 if (bi == 0 and li > 1) else 1
            o = F.relu(bn(F.conv2d(h, sd[pre+".conv1.weight"], stride=stride, padding=1), pre+".bn1"))
            o = bn(F.conv2d(o, sd[pre+".conv2.weight"], padding=1), pre+".bn2")
            sc = bn(F.conv2d(h, sd[pre+".downsample.0.weight"], stride=stride), pre+".downsample.1") if (pre+".downsample.0.weight") in sd else h
            h = F.relu(o + sc)
    return F.linear(F.adaptive_avg_pool2d(h, 1).flatten(1), sd["fc.weight"], sd["fc.bias"])

import os, sys, numpy as np, torch; sys.path.insert(0, os.getcwd()); torch.set_num_threads(8)
import ublib as U; from toyzoo import forward_flat, K
ref=np.load('../dirout/dir_neuron_r1_s0.npz'); z=U.load_zoo(); Msel=ref['models']
M=np.repeat(Msel,K); C=np.tile(np.arange(K),len(Msel)); th0=torch.tensor(z['theta'][-1,M]); c=torch.tensor(C)
rng=np.random.default_rng(7); Xf,yf=U.class_data(rng,C,256); Xh,yh=U.class_data(rng,C,200)
rec=lambda th:(forward_flat(th,Xh).argmax(-1)==yh).float().mean(1)
with torch.no_grad(): h0=rec(th0)
n0=th0.norm(dim=1); th=th0.clone(); done=torch.zeros(len(C),dtype=torch.bool); out=th0.clone(); st=torch.full((len(C),),2000)
for t in range(2001):
    with torch.no_grad(): hit=(~done)&(rec(th)<=U.REL*h0)
    out[hit]=th[hit]; st[hit]=t; done|=hit
    if done.all() or t==2000: break
    x=th.clone().requires_grad_(True)
    L=torch.nn.functional.cross_entropy(forward_flat(x,Xf).reshape(-1,K),yf.reshape(-1),reduction='none').reshape(len(C),-1).mean(1).sum()
    g,=torch.autograd.grad(L,x); u=g/g.norm(dim=1,keepdim=True).clamp_min(1e-12)
    th=(th+u*(U.ETA*n0)[:,None]*(~done)[:,None]).detach()
out[~done]=th[~done]
np.savez_compressed('dir_ga_normstep.npz',d=(out-th0).numpy().astype(np.float32),steps=st.numpy(),stopped=done.numpy())
print('stopped',done.float().mean().item(),'median steps',st.float().median().item())

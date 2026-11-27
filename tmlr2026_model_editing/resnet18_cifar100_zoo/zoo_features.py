"""Baseline features for all models: weight statistics s(W) (62 tensors x 8) and per-class fc-row statistics (100 x 7).
usage: python zoo_features.py ZOO_ROOT -> zoo_features.npz"""
import os, sys, time
sys.path.append(os.getcwd())
import numpy as np, torch
import resnet_zoo as rz
root = sys.argv[1]; zoo = rz.Zoo(root); dev = rz.DEV; T0 = time.time()
ws = np.zeros((zoo.M, rz.N_TENSORS * 8), np.float32); fcs = np.zeros((zoo.M, rz.NC, 7), np.float32); norms = np.zeros(zoo.M, np.float32)
for s in range(0, zoo.M, 20):
    th = zoo.rows(np.arange(s, min(zoo.M, s + 20))).to(dev)
    ws[s:s + len(th)] = rz.weight_statistics(th).cpu().numpy(); fcs[s:s + len(th)] = rz.fc_row_statistics(th).cpu().numpy(); norms[s:s + len(th)] = th.norm(dim=1).cpu().numpy()
    if s % 200 == 0: print(f"[{time.time()-T0:.0f}s] {s}", flush=True)
np.savez("zoo_features.npz", ws=ws, fcs=fcs, theta_norm=norms); print("DONE", ws.shape, fcs.shape, f"{time.time()-T0:.0f}s")

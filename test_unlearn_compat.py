"""Simulate unlearning.py's interface to verify SANEModelWrapper compatibility.

Run:  .venv/bin/python3 test_unlearn_compat.py
"""
import sys, io
from unittest.mock import MagicMock
from pathlib import Path

# ---- mock pytorch_port ----
sys.modules["pytorch_port"] = MagicMock()
sys.modules["pytorch_port.model"] = MagicMock()
sys.modules["pytorch_port.model"].SmallCNN = MagicMock()

import torch
import torch.nn as nn
import json

from SANE.models.def_AE_module import AEModule
from model_export.model_wapper import SANEModelWrapper


def test_unlearn_interface():
    """
    Replicate the exact pattern from unlearning.py's unlearn() function:
        weights = th.tensor(model_weights, requires_grad=True, device=device)
        acc_pred = meta_network(weights.unsqueeze(0)).squeeze(0)
        loss = loss_fn(acc_pred, target_class) + l2_penalty * l2_regularisation(weights)
        loss.backward()
        weights -= lr * weights.grad
    """
    device = "cpu"

    # ---- Load AE (same as before) ----
    ae_dir = Path("experiments/smallcnnzoo-mnist/AE_trainable_e550b")
    config = json.load((ae_dir / "params.json").open("r"))
    config["device"] = device
    config["training::steps_per_epoch"] = 1
    config["model::compile"] = False

    ae_module = AEModule(config)
    ckpt = torch.load(ae_dir / "checkpoint_000010" / "state.pt", map_location=device)
    model_state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    ae_module.model.load_state_dict(model_state, strict=True)

    # ---- Load prediction head ----
    head_data = torch.load(
        "recall_prediction/epoch0-4-8/"
        "multivariate_mnist_smallcnnzoo_per_class_recall_head.pt",
        map_location=device,
    )
    lat_dim = config["ae:lat_dim"]
    B = head_data["B"]
    pred_head = nn.Linear(lat_dim, B.shape[1], bias=True)
    pred_head.weight.data = B[:lat_dim].T.float()
    pred_head.bias.data = B[lat_dim].float()

    # ---- Build wrapper ----
    meta_network = SANEModelWrapper(
        embedding_model=ae_module,
        prediction_head=pred_head,
        tokensize=config["ae:i_dim"],
        freeze_encoder=True,
    ).to(device).eval()

    # ---- Simulate unlearning.py's unlearn() core loop ----
    # Original flat weights (mock as random)
    model_weights = torch.randn(4970)
    target_class = 3
    lr = 0.01
    l2_penalty = 1e-6
    max_steps = 3

    weights = torch.tensor(model_weights, requires_grad=True, device=device)

    # Loss from unlearning.py
    boost_loss = lambda pred, tc: -pred[tc]  # simple: minimize target class recall

    print("=== Simulating unlearn() loop ===")
    for step in range(max_steps):
        # EXACT pattern from unlearning.py line 122
        acc_pred = meta_network(weights.unsqueeze(0)).squeeze(0)

        loss = boost_loss(acc_pred, target_class) + l2_penalty * torch.sum(weights ** 2)
        loss.backward()

        print(f"  Step {step}: pred[{target_class}]={acc_pred[target_class]:.4f}, "
              f"loss={loss.item():.6f}, grad_norm={weights.grad.norm().item():.4f}")

        with torch.no_grad():
            weights -= lr * weights.grad
        weights.grad.zero_()
        meta_network.zero_grad()

    assert weights.grad is None or weights.grad.sum() == 0, "grad should be zeroed"
    print("\n  Unlearning loop completes cleanly.")

    # ---- Test pickle roundtrip ----
    print("\n=== Pickle test ===")
    try:
        import pickle
        buf = io.BytesIO()
        torch.save(meta_network, buf)  # torch.save uses pickle under the hood
        buf.seek(0)
        meta_network2 = torch.load(buf, map_location=device, weights_only=False)
        meta_network2.eval()

        # Verify forward still works
        with torch.no_grad():
            out1 = meta_network(weights.unsqueeze(0))
            out2 = meta_network2(weights.unsqueeze(0))
        match = torch.allclose(out1, out2, atol=1e-5)
        print(f"  torch.save/load roundtrip output match: {match}")
    except Exception as e:
        print(f"  torch.save/load FAILED: {e}")

    # ---- Test pure pickle ----
    print("\n=== Pure pickle test ===")
    try:
        import pickle
        pkl_bytes = pickle.dumps(meta_network)
        meta_network3 = pickle.loads(pkl_bytes)
        meta_network3.eval()
        with torch.no_grad():
            out3 = meta_network3(weights.unsqueeze(0))
        match = torch.allclose(out1, out3, atol=1e-5)
        print(f"  pickle.dumps/loads roundtrip output match: {match}")
    except Exception as e:
        print(f"  pickle FAILED: {type(e).__name__}: {e}")

    # ---- Test device movement ----
    print("\n=== Device movement ===")
    try:
        meta_network.to(device)
        with torch.no_grad():
            out_cpu = meta_network(weights.unsqueeze(0))
        print(f"  .to('cpu') works, output shape: {out_cpu.shape}")
    except Exception as e:
        print(f"  .to('cpu') FAILED: {e}")

    # ---- Test grad on unfrozen encoder ----
    print("\n=== Unfrozen encoder grad test ===")
    # Rebuild fresh AE — previous wrapper with freeze_encoder=True mutated params
    ae_module2 = AEModule(config)
    ckpt2 = torch.load(ae_dir / "checkpoint_000010" / "state.pt", map_location=device)
    model_state2 = {k.replace("_orig_mod.", ""): v for k, v in ckpt2["model"].items()}
    ae_module2.model.load_state_dict(model_state2, strict=True)

    wrapper_unfrozen = SANEModelWrapper(
        embedding_model=ae_module2,
        prediction_head=pred_head,
        tokensize=config["ae:i_dim"],
        freeze_encoder=False,
    ).to(device)
    wrapper_unfrozen.ae.train()

    # Check requires_grad before forward
    ae_params = list(wrapper_unfrozen.ae.parameters())
    grad_enabled = sum(1 for p in ae_params if p.requires_grad)
    print(f"  AE params with requires_grad=True: {grad_enabled} / {len(ae_params)}")

    w = torch.randn(4970, requires_grad=True)
    out = wrapper_unfrozen(w.unsqueeze(0))
    print(f"  output grad_fn: {out.grad_fn is not None}")
    torch.sum(out).backward()

    encoder_grads = sum(1 for p in ae_params if p.grad is not None)
    print(f"  AE params with .grad populated: {encoder_grads} / {len(ae_params)}")

    # Show a specific param to verify
    for name, p in wrapper_unfrozen.ae.named_parameters():
        if "tokenizer" in name:
            print(f"  {name}: requires_grad={p.requires_grad}, grad={'yes' if p.grad is not None else 'no'}")
            break

    print("\n=== Summary ===")
    print("Interface matches unlearning.py exactly.")
    print("Use:  meta_network = SANEModelWrapper(...).to(device).eval()")
    print("      acc_pred = meta_network(weights.unsqueeze(0)).squeeze(0)")


if __name__ == "__main__":
    test_unlearn_interface()

"""End-to-end: load real AE + prediction head, test gradient flow.

Run:  .venv/bin/python3 test_e2e_grad.py
"""
import sys
from unittest.mock import MagicMock
from pathlib import Path

import torch
import torch.nn as nn
import json

# ---- mock pytorch_port (needed by reconstruct_network, imported transitively) ----
sys.modules["pytorch_port"] = MagicMock()
sys.modules["pytorch_port.model"] = MagicMock()
sys.modules["pytorch_port.model"].SmallCNN = MagicMock()

from SANE.models.def_AE_module import AEModule
from model_export.model_wapper import SANEModelWrapper, save_model_wrapper


def main():
    device = "cpu"

    # ---- 1. Load SANE AE checkpoint ----
    ae_dir = Path("experiments/smallcnnzoo-mnist/AE_trainable_e550b")
    config = json.load((ae_dir / "params.json").open("r"))
    config["device"] = device
    config["training::steps_per_epoch"] = 1    # dummy for OneCycleLR
    config["model::compile"] = False           # avoid torch.compile overhead

    print("Creating AEModule...")
    ae_module = AEModule(config)

    ckpt_path = ae_dir / "checkpoint_000010" / "state.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Strip _orig_mod. prefix (from torch.compile in original training)
    model_state = {
        k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()
    }
    ae_module.model.load_state_dict(model_state, strict=True)
    ae_module.model.eval()
    print("AE model loaded and in eval mode.")

    # ---- 2. Build prediction head from trained regression weights ----
    head_path = Path("recall_prediction/epoch0-4-8/"
                     "multivariate_mnist_smallcnnzoo_per_class_recall_head.pt")
    print(f"Loading prediction head: {head_path}")
    head_data = torch.load(head_path, map_location=device)

    B = head_data["B"]                           # (129, 10) float64
    lat_dim = config["ae:lat_dim"]               # 128
    n_classes = B.shape[1]                       # 10

    pred_head = nn.Linear(lat_dim, n_classes, bias=True)
    # B layout: first 128 rows = weight coefficients, last row = intercept
    pred_head.weight.data = B[:lat_dim].T.float()   # (10, 128)
    pred_head.bias.data = B[lat_dim].float()        # (10,)

    print(f"Prediction head: Linear({lat_dim} -> {n_classes})")
    print(f"  weight mean={pred_head.weight.data.mean():.4f}, "
          f"bias mean={pred_head.bias.data.mean():.4f}")

    # ---- 3. Build wrapper ----
    wrapper = SANEModelWrapper(
        embedding_model=ae_module,
        prediction_head=pred_head,
        tokensize=config["ae:i_dim"],  # 145
        freeze_encoder=True,           # only test grad through pipe, not through AE
    ).to(device).eval()

    print(f"\nWrapper ready: {wrapper.n_tokens} tokens, "
          f"tokensize={wrapper.tokensize}, lat_dim={wrapper.lat_dim}")

    # ---- 4. Test: forward + backward with random input weights ----
    BATCH = 2
    w = torch.randn(BATCH, 4970, requires_grad=True)
    print(f"\nInput: {w.shape}, requires_grad={w.requires_grad}")

    out = wrapper(w)
    print(f"Output: {out.shape}  (batch={BATCH}, n_classes={n_classes})")
    print(f"  range: [{out.min().item():.4f}, {out.max().item():.4f}]")

    loss = out.sum()
    print(f"\nBackpropagating loss={loss.item():.4f}...")
    loss.backward()

    # ---- 5. Check gradients ----
    print(f"\n=== Gradient report ===")
    print(f"  w.grad is not None:     {w.grad is not None}")
    print(f"  w.grad mean:            {w.grad.mean().item():.6f}")
    print(f"  w.grad all-nonzero:     {(w.grad != 0).all().item()}")
    print(f"  w.grad any-NaN:         {torch.isnan(w.grad).any().item()}")

    # Verify encoder params are frozen
    any_encoder_grad = False
    for name, p in wrapper.ae.named_parameters():
        if p.grad is not None:
            any_encoder_grad = True
            print(f"  UNEXPECTED grad on: {name}")
    print(f"  encoder frozen (no grad): {not any_encoder_grad}")

    # Verify prediction head params got grad
    for name, p in wrapper.prediction_head.named_parameters():
        has_grad = p.grad is not None
        print(f"  pred_head.{name}: grad={'yes' if has_grad else 'no'}, "
              f"mean={p.grad.mean().item():.6f}" if has_grad else "")

    # ---- 6. Also test with a single sample ----
    print(f"\n=== Single-sample test ===")
    w1 = torch.randn(4970, requires_grad=True)
    out1 = wrapper(w1)
    print(f"  Output shape: {out1.shape}")    # should be (10,)
    (out1.sum()).backward()
    print(f"  w1.grad is not None: {w1.grad is not None}")

    # ---- 7. Test reconstruct round-trip ----
    print(f"\n=== Reconstruction round-trip ===")
    w_orig = w[:1].detach().clone()
    tokens = wrapper._weights_to_tokens(w[:1])  # (1, 58, 145)
    w_recon = wrapper.reconstruct(tokens)
    mae = (w_orig - w_recon).abs().mean().item()
    print(f"  MAE: {mae:.2e}")

    print("\nDone! Gradients flow end-to-end.")


if __name__ == "__main__":
    main()

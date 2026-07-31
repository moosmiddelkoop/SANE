"""Integration tests for SANEModelWrapper.

Run:  .venv/bin/python3 test_model_wrapper.py
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

# ---- mock pytorch_port BEFORE anything imports it ----
sys.modules["pytorch_port"] = MagicMock()
sys.modules["pytorch_port.model"] = MagicMock()
sys.modules["pytorch_port.model"].SmallCNN = MagicMock()

# Simulate SANE packages that may not be installed
sys.modules["SANE"] = MagicMock()
sys.modules["SANE.models"] = MagicMock()
sys.modules["SANE.models.def_AE"] = MagicMock()
sys.modules["SANE.models.def_AE_module"] = MagicMock()
sys.modules["SANE.models.def_loss"] = MagicMock()
sys.modules["SANE.models.def_transformer"] = MagicMock()

from model_export.reconstruct_network import TOTAL_PARAMS
from model_export.diff_weight_pipe import (
    flat_to_state_dict,
    tokenize,
    detokenize,
    state_dict_to_flat,
    REF_SHAPES,
)
from model_export.model_wapper import (
    SANEModelWrapper,
    save_model_wrapper,
)


# ------------------------------------------------------------------
# Mock AEModule for testing without full SANE install
# ------------------------------------------------------------------
class MockAEModule(nn.Module):
    """Stand-in for AEModule — exposes forward_embeddings + config."""

    def __init__(self, lat_dim=128, i_dim=289, windowsize=93):
        super().__init__()
        self.config = {
            "ae:lat_dim": lat_dim,
            "ae:i_dim": i_dim,
            "ae:d_model": 512,
            "ae:nhead": 8,
            "ae:num_layers": 4,
            "ae:max_positions": [100, 10, 40],
            "ae:transformer_type": "gpt2",
            "training::windowsize": windowsize,
            "model::compile": False,
            "training::steps_per_epoch": 1,
            "training::epochs_train": 50,
            "training::test_epochs": 1,
            "optim::optimizer": "adamw",
            "optim::lr": 1e-4,
        }
        # Simple encoder: Linear(token_dim → lat_dim) then mean-pool over seq
        self.encoder = nn.Sequential(
            nn.Linear(i_dim, 64),
            nn.ReLU(),
            nn.Linear(64, lat_dim),
        )

    def forward_embeddings(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """x: (B, seq, token_dim), p: (B, seq, 3) — ignored but checked."""
        # Verify shapes
        assert x.ndim == 3, f"Expected 3D input, got {x.ndim}D"
        assert p.ndim == 3, f"Expected 3D pos, got {p.ndim}D"
        # Encode and mean-pool
        z = self.encoder(x)      # (B, seq, lat_dim)
        return z.mean(dim=1)     # (B, lat_dim)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
class TestSANEModelWrapper(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.lat_dim = 128
        cls.i_dim = 289
        cls.n_classes = 10
        cls.batch_size = 4

    def setUp(self):
        self.ae = MockAEModule(lat_dim=self.lat_dim, i_dim=self.i_dim)
        self.pred_head = nn.Linear(self.lat_dim, self.n_classes)
        self.wrapper = SANEModelWrapper(
            embedding_model=self.ae,
            prediction_head=self.pred_head,
            tokensize=self.i_dim,
        )

    # ---- shape / structure tests ----

    def test_n_tokens_matches(self):
        """Wrapper pre-computes correct token count for SmallCNN."""
        sd = flat_to_state_dict(torch.randn(TOTAL_PARAMS))
        tokens, _mask, pos = tokenize(sd, tokensize=self.i_dim)
        self.assertEqual(self.wrapper.n_tokens, tokens.shape[0])
        self.assertEqual(self.wrapper.n_tokens, 58)

    def test_pos_match_between_fixed_and_runtime(self):
        """The pre-computed _pos matches the pos from runtime tokenize."""
        w = torch.randn(TOTAL_PARAMS)
        sd = flat_to_state_dict(w)
        _t, _m, runtime_pos = tokenize(sd, tokensize=self.i_dim)
        self.assertTrue(torch.equal(self.wrapper._pos, runtime_pos))

    def test_forward_shape(self):
        """Forward returns (batch, n_classes)."""
        w = torch.randn(self.batch_size, TOTAL_PARAMS)
        out = self.wrapper(w)
        self.assertEqual(out.shape, (self.batch_size, self.n_classes))

    def test_forward_single_sample(self):
        """Forward accepts 1D input and returns 1D output."""
        w = torch.randn(TOTAL_PARAMS)
        out = self.wrapper(w)
        self.assertEqual(out.shape, (self.n_classes,))

    def test_encode_shape(self):
        """Encode returns (batch, lat_dim)."""
        w = torch.randn(self.batch_size, TOTAL_PARAMS)
        emb = self.wrapper.encode(w)
        self.assertEqual(emb.shape, (self.batch_size, self.lat_dim))

    # ---- gradient tests ----

    def test_grad_flows_to_weights(self):
        """Grad flows from prediction loss to input weights."""
        w = torch.randn(self.batch_size, TOTAL_PARAMS, requires_grad=True)
        out = self.wrapper(w)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(w.grad)
        self.assertFalse((w.grad == 0).all(),
                         "weight grad should be non-zero everywhere")

    def test_grad_flows_with_frozen_encoder(self):
        """With freeze_encoder=True, weight grad still flows but encoder
        params get no grad."""
        wrapper_frozen = SANEModelWrapper(
            embedding_model=self.ae,
            prediction_head=self.pred_head,
            tokensize=self.i_dim,
            freeze_encoder=True,
        )
        w = torch.randn(1, TOTAL_PARAMS, requires_grad=True)
        out = wrapper_frozen(w)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(w.grad)
        for name, p in wrapper_frozen.ae.named_parameters():
            self.assertEqual(p.grad, None,
                             f"{name}: frozen encoder should have no grad")

    def test_grad_through_reconstruct(self):
        """Grad flows from reconstructed flat back to tokens."""
        tokens = torch.randn(self.batch_size, self.wrapper.n_tokens,
                             self.i_dim, requires_grad=True)
        flats = self.wrapper.reconstruct(tokens)
        loss = flats.sum()
        loss.backward()

        self.assertIsNotNone(tokens.grad)
        self.assertFalse((tokens.grad == 0).all())

    def test_grad_through_multiple_forward(self):
        """Multiple forward passes accumulate correctly (no leak)."""
        self.wrapper.eval()  # disable dropout for deterministic pass
        w = torch.randn(self.batch_size, TOTAL_PARAMS, requires_grad=True)

        out1 = self.wrapper(w)
        loss1 = out1.sum()
        loss1.backward()
        grad1 = w.grad.clone()

        w.grad = None
        out2 = self.wrapper(w)
        loss2 = (out2 * 2).sum()
        loss2.backward()
        grad2 = w.grad.clone()

        self.assertTrue(torch.allclose(grad2, grad1 * 2, atol=1e-4),
                        "grad should scale with loss factor")

    # ---- reconstruction tests ----

    def test_reconstruct_roundtrip(self):
        """flat → tokens → flat should reconstruct exactly."""
        w = torch.randn(TOTAL_PARAMS)
        sd = flat_to_state_dict(w)
        tokens, _mask, pos = tokenize(sd, tokensize=self.i_dim)
        tokens_batch = tokens.unsqueeze(0)
        w2 = self.wrapper.reconstruct(tokens_batch).squeeze(0)
        self.assertTrue(torch.allclose(w, w2, atol=1e-5))

    # ---- save / load tests ----

    def test_save_load_roundtrip(self):
        """Save a checkpoint → manually reload → forward produces identical output."""
        self.wrapper.eval()  # disable dropout for deterministic comparison
        w = torch.randn(self.batch_size, TOTAL_PARAMS)
        orig_out = self.wrapper(w).detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model_wrapper(self.wrapper, tmpdir)

            ckpt = torch.load(Path(tmpdir) / "model.pt")
            loaded_ae = MockAEModule(lat_dim=self.lat_dim, i_dim=self.i_dim)
            loaded_ae.load_state_dict(ckpt["encoder"], strict=True)
            loaded_head = nn.Linear(self.lat_dim, self.n_classes)
            loaded_head.load_state_dict(ckpt["prediction_head"], strict=True)

            loaded = SANEModelWrapper(
                embedding_model=loaded_ae,
                prediction_head=loaded_head,
                tokensize=self.i_dim,
            )
            loaded.eval()

            loaded_out = loaded(w).detach().clone()
            self.assertTrue(torch.allclose(orig_out, loaded_out, atol=1e-6))
            checkpoint = torch.load(Path(tmpdir) / "model.pt")
            loaded.ae.load_state_dict(
                checkpoint["encoder"], strict=False
            )
            loaded.prediction_head.load_state_dict(
                checkpoint["prediction_head"]
            )

            loaded_out = loaded(w).detach().clone()
            self.assertTrue(torch.allclose(orig_out, loaded_out, atol=1e-6))

    # ---- edge cases ----

    def test_works_with_numpy_input(self):
        """Converting numpy array manually still works (wrapper expects tensor)."""
        arr = np.random.randn(TOTAL_PARAMS).astype(np.float32)
        t = torch.from_numpy(arr).float()
        out = self.wrapper(t.unsqueeze(0))
        self.assertEqual(out.shape, (1, self.n_classes))

    def test_empty_batch_raises(self):
        """Empty batch should still produce correct shapes."""
        w = torch.randn(0, TOTAL_PARAMS)
        out = self.wrapper(w)
        self.assertEqual(out.shape, (0, self.n_classes))


if __name__ == "__main__":
    unittest.main(verbosity=2)

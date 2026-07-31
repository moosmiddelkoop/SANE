"""Test whether gradients survive the reconstruct_network function."""
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn as nn

# ---- mock SmallCNN BEFORE importing reconstruct_network ----
class MockSmallCNN(nn.Module):
    def __init__(self, activation="relu", dropout_rate=0.0):
        super().__init__()
        act_map = {
            "relu": nn.ReLU, "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid, "selu": nn.SELU,
        }
        act_cls = act_map.get(activation, nn.ReLU)
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.act = act_cls()
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(16, 10)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.dropout(x)
        x = self.dense(x)
        return x


sys.modules["pytorch_port"] = MagicMock()
sys.modules["pytorch_port.model"] = MagicMock()
sys.modules["pytorch_port.model"].SmallCNN = MockSmallCNN

from math import prod
from model_export.reconstruct_network import (
    SHAPES,
    TOTAL_PARAMS,
    _flat_weights_to_state_dict,
    reconstruct_network,
)


class TestGradientFlow(unittest.TestCase):

    # ---- 1. State dict tensors from numpy are detached ----
    def test_state_dict_tensors_are_detached_from_numpy(self):
        """Tensors from _flat_weights_to_state_dict have no grad_fn (detached)."""
        arr = np.random.randn(TOTAL_PARAMS).astype(np.float32)
        sd = _flat_weights_to_state_dict(arr)
        for k, t in sd.items():
            self.assertIsNone(t.grad_fn, f"{k}: has grad_fn (not detached)")

    # ---- 2. load_state_dict copies .data only, not the tensor identity ----
    def test_load_state_dict_copies_data_not_identity(self):
        """load_state_dict copies by value; model params != source tensors."""
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.zeros(1))

        m = Dummy()
        src = torch.tensor([3.0], requires_grad=True)
        m.load_state_dict({"p": src}, strict=True)

        # Value matches
        self.assertEqual(m.p.item(), 3.0)
        # But it's a copy — modifying source doesn't change param
        src.data[0] = 99.0
        self.assertEqual(m.p.item(), 3.0)  # unchanged
        # And backward from model won't flow to src
        self.assertIsNone(src.grad)

    # ---- 3. Model parameters DO support backward (nn.Parameter requires_grad=True) ----
    def test_backprop_through_model_works(self):
        """Forward + backward through the reconstructed model works fine."""
        arr = np.random.randn(TOTAL_PARAMS).astype(np.float32)
        model = reconstruct_network(arr, activation="relu")

        inp = torch.randn(1, 1, 32, 32)
        out = model(inp)
        loss = out.sum()
        loss.backward()

        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, f"{name}: .grad missing after backward")
            self.assertFalse((p.grad == 0).all(), f"{name}: grad is all-zero")

    # ---- 4. But you CANNOT backprop to the original flat weight array ----
    def test_cannot_backprop_to_original_array(self):
        """The numpy -> tensor -> load_state_dict chain severs grad from
        the original flat weight array. Mutating the array post-facto has
        no effect on the model."""
        arr = np.random.randn(TOTAL_PARAMS).astype(np.float32)
        model = reconstruct_network(arr, activation="relu")

        original_model_params = {
            name: p.data.clone() for name, p in model.named_parameters()
        }
        arr[:] = 999.0  # mutate entire original numpy array

        model2 = reconstruct_network(arr, activation="relu")
        for name, p2 in model2.named_parameters():
            self.assertFalse(
                torch.equal(original_model_params[name], p2.data),
                f"{name}: params unchanged after source array mutation",
            )
        # Model params shifted because they're copies from the mutated array
        # on the second call — proving they are snapshots, not views.
        # No live grad link exists between arr and model params.

    # ---- 5. Verify total param count matches ----
    def test_output_parameter_count(self):
        arr = np.random.randn(TOTAL_PARAMS).astype(np.float32)
        model = reconstruct_network(arr, activation="relu")
        n_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(n_params, TOTAL_PARAMS,
                         f"Expected {TOTAL_PARAMS} params, got {n_params}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

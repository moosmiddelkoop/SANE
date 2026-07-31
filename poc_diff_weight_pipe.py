"""
PoC: Fully differentiable pipeline: flat weight vector <-> state dict <-> tokens.

All ops are torch-native — no numpy, no .data.copy_, no deepcopy.
Gradients flow end-to-end from reconstructed weights back to the source vector.

Run:  .venv/bin/python3 poc_diff_weight_pipe.py
"""
import unittest
import sys
from unittest.mock import MagicMock

import torch

# ---- mock pytorch_port (not needed for this PoC) ----
sys.modules["pytorch_port"] = MagicMock()
sys.modules["pytorch_port.model"] = MagicMock()
sys.modules["pytorch_port.model"].SmallCNN = MagicMock()

from model_export.reconstruct_network import SHAPES, TOTAL_PARAMS


class DiffWeightPipe:
    """Fully differentiable flat-weight <-> tokens pipeline for SmallCNN.

    Uses the same SHAPES / TF-storage-order convention as model_export.
    """

    # TF-name → (pt_key, forward_transpose_axes, is_conv)
    _TF_TO_PT = [
        # (tf_name,                       pt_key,          tf->pt transpose)
        ("sequential/conv2d/bias:0",      "conv1.bias",    None),
        ("sequential/conv2d/kernel:0",    "conv1.weight",  (3, 2, 0, 1)),  # kH,kW,C_in,C_out → C_out,C_in,kH,kW
        ("sequential/conv2d_1/bias:0",    "conv2.bias",    None),
        ("sequential/conv2d_1/kernel:0",  "conv2.weight",  (3, 2, 0, 1)),
        ("sequential/conv2d_2/bias:0",    "conv3.bias",    None),
        ("sequential/conv2d_2/kernel:0",  "conv3.weight",  (3, 2, 0, 1)),
        ("sequential/dense/bias:0",       "dense.bias",    None),
        ("sequential/dense/kernel:0",     "dense.weight",  (1, 0)),         # in,out → out,in
    ]

    # Subset of PT keys that have both weight + bias (token layout = [weight|bias])
    _WEIGHT_KEYS = {"conv1.weight", "conv2.weight", "conv3.weight", "dense.weight"}
    _BIAS_KEYS   = {"conv1.bias", "conv2.bias", "conv3.bias", "dense.bias"}

    # Layer ordering for position encoding
    _LAYER_ORDER = ["conv1", "conv2", "conv3", "dense"]

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # 1.  flat  →  state dict  (differentiable)
    # ------------------------------------------------------------------
    @classmethod
    def flat_to_state_dict(cls, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convert a flat tensor (TF storage order) to a PyTorch state dict.

        Args:
            flat: 1-D tensor of length TOTAL_PARAMS.  Must preserve any grad graph.

        Returns:
            dict mapping PT keys to tensors.  All tensors share the flat's grad graph.
        """
        assert flat.shape == (TOTAL_PARAMS,), f"Expected ({TOTAL_PARAMS},), got {flat.shape}"

        sd = {}
        offset = 0
        for tf_name, pt_key, transpose in cls._TF_TO_PT:
            tf_shape = SHAPES[tf_name]
            n = 1
            for d in tf_shape:
                n *= d
            chunk = flat[offset : offset + n].reshape(tf_shape)
            if transpose is not None:
                chunk = chunk.permute(*transpose)
            sd[pt_key] = chunk
            offset += n
        return sd

    # ------------------------------------------------------------------
    # 2.  state dict  →  flat  (differentiable)
    # ------------------------------------------------------------------
    @classmethod
    def state_dict_to_flat(cls, sd: dict[str, torch.Tensor]) -> torch.Tensor:
        """Reverse of flat_to_state_dict: PT state dict → TF-order flat tensor.

        Args:
            sd: PT state dict (tensors may have grad graph).

        Returns:
            1-D tensor of length TOTAL_PARAMS.
        """
        pieces = []
        for tf_name, pt_key, transpose in cls._TF_TO_PT:
            t = sd[pt_key]
            if transpose is not None:
                # invert the forward transpose
                inv = tuple(torch.argsort(torch.tensor(transpose)).tolist())
                t = t.permute(*inv)
            # t is now in TF shape; flatten
            pieces.append(t.reshape(-1))
        return torch.cat(pieces, dim=0)

    # ------------------------------------------------------------------
    # 3a.  state dict  →  tokens  (differentiable)
    # ------------------------------------------------------------------
    @classmethod
    def tokenize(
        cls, sd: dict[str, torch.Tensor], tokensize: int = 0, ignore_bn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize a state dict into (tokens, mask, pos).  Grad-safe.

        Mimics tokenize_checkpoint but with guaranteed grad flow.
        Only handles layers without BN (SmallCNN has none).

        Returns:
            tokens:  (N_tokens, tokensize)  float
            mask:    (N_tokens, tokensize)  bool
            pos:     (N_tokens, 3)          int  [global_idx, layer_idx, chan_idx]
        """
        # --- discover tokensize ---
        if tokensize == 0:
            tokensize = 0
            for key in sd:
                if ignore_bn and ("bn" in key or "downsample.1" in key):
                    continue
                if "weight" not in key and "running_mean" not in key and "running_var" not in key:
                    continue
                w = sd[key]
                per_chan = w.numel() // w.shape[0]
                if key.replace("weight", "bias") in sd:
                    per_chan += 1
                tokensize = max(tokensize, per_chan)
        tokensize = int(tokensize)

        # --- build tokens layer by layer ---
        all_tokens = []
        all_masks = []
        all_pos = []
        global_idx = 0

        for lidx, lname in enumerate(cls._LAYER_ORDER):
            wkey = f"{lname}.weight"
            bkey = f"{lname}.bias"
            if wkey not in sd:
                continue

            w = sd[wkey]                              # (C_out, ...)
            w = w.reshape(w.shape[0], -1)             # (C_out, per_chan_weight)
            if bkey in sd:
                b = sd[bkey]                          # (C_out,)
                w = torch.cat([w, b.unsqueeze(1)], dim=1)  # (C_out, per_chan_weight + 1)

            C_out = w.shape[0]
            per_chan = w.shape[1]                     # includes bias column

            # How many token-slots per channel?
            a = per_chan // tokensize
            b_rem = per_chan % tokensize
            token_factor = a + (1 if b_rem > 0 else 0)

            # --- zero-pad if needed ---
            padded_len = tokensize * token_factor
            if b_rem > 0:
                w_padded = torch.zeros(C_out, padded_len, device=w.device, dtype=w.dtype)
                w_padded[:, :per_chan] = w
                mask_layer = torch.zeros(C_out, padded_len, device=w.device, dtype=torch.bool)
                mask_layer[:, :per_chan] = True
            else:
                w_padded = w
                mask_layer = torch.ones(C_out, padded_len, device=w.device, dtype=torch.bool)

            # --- reshape to tokens: (C_out * token_factor, tokensize) ---
            w_tokens = w_padded.reshape(-1, tokensize)       # grad-safe
            mask_tokens = mask_layer.reshape(-1, tokensize)  # bool

            all_tokens.append(w_tokens)
            all_masks.append(mask_tokens)

            # --- positions ---
            for c in range(C_out):
                for _ in range(token_factor):
                    all_pos.append((global_idx, lidx, c))
                    global_idx += 1

        tokens = torch.cat(all_tokens, dim=0)
        mask = torch.cat(all_masks, dim=0)
        pos = torch.tensor(all_pos, dtype=torch.int)

        return tokens, mask, pos

    # ------------------------------------------------------------------
    # 3b.  tokens  →  state dict  (differentiable)
    # ------------------------------------------------------------------
    @classmethod
    def detokenize(
        cls, tokens: torch.Tensor, pos: torch.Tensor, ref_shapes: dict,
    ) -> dict[str, torch.Tensor]:
        """Convert tokens back to a state dict.  Grad-safe.

        Uses pos[:, 1] (layer index) to group tokens, then reshapes back
        to the original parameter shapes stored in ref_shapes.

        Args:
            tokens:     (N, tokensize)  float  (may have grad graph)
            pos:        (N, 3)          int    layer-membership info
            ref_shapes: dict mapping PT keys → original torch.Size

        Returns:
            State dict with tensors whose values depend on tokens.
            Grad flows through index_select + view + slice.
        """
        sd = {}

        for lidx, lname in enumerate(cls._LAYER_ORDER):
            wkey = f"{lname}.weight"
            bkey = f"{lname}.bias"
            if wkey not in ref_shapes:
                continue

            # select tokens for this layer
            layer_indices = torch.where(pos[:, 1] == lidx)[0]
            layer_tokens = torch.index_select(tokens, dim=0, index=layer_indices)  # (M, tokensize)

            w_shape = ref_shapes[wkey]            # e.g. (16, 1, 3, 3)
            C_out = w_shape[0]
            per_chan = w_shape.numel() // C_out    # e.g. 9
            if bkey in ref_shapes:
                per_chan += 1                      # +1 for bias column

            # reshape to (C_out, token_factor * tokensize), then strip padding
            flat_per_channel = layer_tokens.reshape(C_out, -1)     # (C_out, token_factor * tokensize)
            chan_content = flat_per_channel[:, :per_chan]          # (C_out, per_chan)

            # split weight from bias
            w_flat = chan_content[:, :per_chan - 1]                # (C_out, per_chan_weight)
            weight = w_flat.reshape(w_shape)

            sd[wkey] = weight
            if bkey in ref_shapes:
                bias = chan_content[:, per_chan - 1]               # (C_out,)
                sd[bkey] = bias

        return sd


# ====================================================================
# PoC tests
# ====================================================================
class TestDiffWeightPipe(unittest.TestCase):

    def setUp(self):
        self.pipe = DiffWeightPipe()
        self.device = torch.device("cpu")
        # Build ref_shapes from a dummy state dict
        flat_ref = torch.randn(TOTAL_PARAMS)
        self.ref_sd = self.pipe.flat_to_state_dict(flat_ref)
        self.ref_shapes = {k: v.shape for k, v in self.ref_sd.items()}

    def test_roundtrip_flat_state_dict(self):
        """flat → sd → flat: values must match exactly."""
        x = torch.randn(TOTAL_PARAMS, requires_grad=False)
        sd = self.pipe.flat_to_state_dict(x)
        y = self.pipe.state_dict_to_flat(sd)
        self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_roundtrip_state_dict_tokens(self):
        """sd → tokens → sd: values must match (ignoring padding)."""
        x = torch.randn(TOTAL_PARAMS)
        sd = self.pipe.flat_to_state_dict(x)
        tokens, mask, pos = self.pipe.tokenize(sd, tokensize=0)
        sd2 = self.pipe.detokenize(tokens, pos, self.ref_shapes)
        for k in sd:
            self.assertTrue(torch.allclose(sd[k], sd2[k], atol=1e-5),
                            f"{k}: roundtrip mismatch")

    def test_full_pipe_roundtrip(self):
        """flat → sd → tokens → sd → flat: must match."""
        x = torch.randn(TOTAL_PARAMS)
        sd1 = self.pipe.flat_to_state_dict(x)
        tokens, mask, pos = self.pipe.tokenize(sd1, tokensize=0)
        sd2 = self.pipe.detokenize(tokens, pos, self.ref_shapes)
        y = self.pipe.state_dict_to_flat(sd2)
        self.assertTrue(torch.allclose(x, y, atol=1e-5))

    # ---- gradient tests ----

    def test_grad_flows_flat_to_tokens(self):
        """Grad from tokens should reach the original flat tensor."""
        flat = torch.randn(TOTAL_PARAMS, requires_grad=True)
        sd = self.pipe.flat_to_state_dict(flat)
        tokens, mask, pos = self.pipe.tokenize(sd, tokensize=0)

        loss = tokens.sum()
        loss.backward()

        self.assertIsNotNone(flat.grad)
        self.assertFalse((flat.grad == 0).all(), "flat.grad is all-zero")

    def test_grad_flows_tokens_to_flat(self):
        """Grad from reconstructed flat should reach tokens."""
        tokens = torch.randn(58, 145, requires_grad=True)
        pos = self._build_pos()
        sd = self.pipe.detokenize(tokens, pos, self.ref_shapes)
        flat = self.pipe.state_dict_to_flat(sd)

        loss = flat.sum()
        loss.backward()

        self.assertIsNotNone(tokens.grad)
        self.assertFalse((tokens.grad == 0).all(), "tokens.grad is all-zero")

    def test_grad_flows_end_to_end(self):
        """Full pipe: grad from reconstructed flat reaches original flat."""
        flat = torch.randn(TOTAL_PARAMS, requires_grad=True)
        sd = self.pipe.flat_to_state_dict(flat)
        tokens, mask, pos = self.pipe.tokenize(sd, tokensize=0)

        # --- simulate SANE: pass tokens through a linear layer ---
        # In reality this would be the AE encoder/decoder, but any
        # differentiable transform proves the point.
        transformed_tokens = tokens * 2.0 + 1.0

        sd2 = self.pipe.detokenize(transformed_tokens, pos, self.ref_shapes)
        flat2 = self.pipe.state_dict_to_flat(sd2)

        loss = flat2.sum()
        loss.backward()

        self.assertIsNotNone(flat.grad)
        # flat2 = state_dict_to_flat(detokenize(2*tokens + 1))
        # grad should be 2.0 for every element (chain: 1 * 2 * dflat2/dtokens * dtokens/dflat)
        # The pipe is identity-ish apart from the ×2+1 transform, so grad ≈ 2.0
        self.assertTrue(torch.allclose(flat.grad, torch.full_like(flat.grad, 2.0), atol=1e-5),
                        f"Expected grad≈2.0, got range [{flat.grad.min():.4f}, {flat.grad.max():.4f}]")

    def test_grad_through_masked_tokens(self):
        """Verify padding regions have zeros and don't corrupt gradients."""
        flat = torch.randn(TOTAL_PARAMS, requires_grad=True)
        sd = self.pipe.flat_to_state_dict(flat)
        tokens, mask, pos = self.pipe.tokenize(sd, tokensize=0)

        # Select a padded element (e.g., dense token 0, position 17)
        dense_token0 = tokens[48]  # tokens before dense: 16+16+16=48
        # positions 0-15 are weight, 16 is bias, 17-144 are padding
        self.assertFalse(mask[48, 17].item(), "column 17 of dense token should be padding")

        # Zero out all padding-region values (they started as 0 but double-check)
        padding_val = tokens[~mask].clone()
        self.assertTrue((padding_val == 0).all(), "padding should be zero")

        # Forward + backward
        # Apply transformation only on non-padded elements
        transformed = tokens * mask.float() * 3.0
        sd2 = self.pipe.detokenize(transformed, pos, self.ref_shapes)
        flat2 = self.pipe.state_dict_to_flat(sd2)
        loss = flat2.sum()
        loss.backward()

        self.assertIsNotNone(flat.grad)
        self.assertFalse(torch.isnan(flat.grad).any(), "grad contains NaNs")
        self.assertTrue((flat.grad != 0).all(), "all elements should have non-zero grad")

    def test_conv_weight_transpose_preserves_grad(self):
        """The PT↔TF transpose in conv layers must be grad-preserving."""
        # TF shape: (3,3,1,16)
        tf_tensor = torch.randn(3, 3, 1, 16, requires_grad=True)
        # forward transpose: TF → PT via (3,2,0,1) → (16,1,3,3)
        pt_tensor = tf_tensor.permute(3, 2, 0, 1)
        # reverse transpose: PT → TF via inverse (2,3,1,0) → (3,3,1,16)
        inv = (2, 3, 1, 0)
        tf2 = pt_tensor.permute(*inv)

        loss = tf2.sum()
        loss.backward()
        self.assertTrue(torch.allclose(tf_tensor.grad, torch.ones_like(tf_tensor)))

    # ---- helpers ----

    def _build_pos(self):
        """Reconstruct pos for testing grad from tokens→flat in isolation."""
        pos = []
        gidx = 0
        for lidx in range(4):
            n_chan = 16 if lidx < 3 else 10
            token_factor = 1  # all layers fit in 1 slot with tokensize=145
            for c in range(n_chan):
                for _ in range(token_factor):
                    pos.append((gidx, lidx, c))
                    gidx += 1
        return torch.tensor(pos, dtype=torch.int)


if __name__ == "__main__":
    unittest.main(verbosity=2)

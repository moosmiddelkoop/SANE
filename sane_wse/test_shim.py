"""Check the sane_wse token plan against model_export.diff_weight_pipe, the pipeline it replaces.

The plan must reproduce `diff_weight_pipe.tokenize` bit-for-bit on the SmallCNN, otherwise existing
unlearning results stop being comparable. No SANE artifacts are needed: the plan depends on shapes
alone, so the encoder and the head are left out.

Run from a repo root that has model_export/:
    .venv/bin/python3 test_sane_wse.py
"""

import sys
from unittest.mock import MagicMock

import torch
from torch import nn

# model_export.reconstruct_network imports pytorch_port, which imports keras. Only its shape
# constants are needed here, so the import is stubbed out, as test_model_wrapper.py does.
sys.modules["pytorch_port"] = MagicMock()
sys.modules["pytorch_port.model"] = MagicMock()

from model_export.diff_weight_pipe import REF_SHAPES, standardize_state_dict, tokenize  # noqa: E402
from sane_wse import SANELens

TOKENSIZE = 145  # = config["ae:i_dim"] of the smallcnnzoo-mnist trial
LAYERS = ["conv1", "conv2", "conv3", "dense"]
STATS = {
    "conv1.weight": {"mean": 0.013, "std": 0.271},
    "conv2.weight": {"mean": -0.004, "std": 0.152},
    "conv3.weight": {"mean": 0.002, "std": 0.119},
    "dense.weight": {"mean": -0.021, "std": 0.334},
}

# native torch state_dict order: weight before bias, which is the order wse hands to prepare()
SHAPES = {f"{layer}.{part}": REF_SHAPES[f"{layer}.{part}"] for layer in LAYERS for part in ("weight", "bias")}


def make_lens(standardization):
    lens = SANELens(
        encoder=nn.Identity(),
        head=nn.Linear(128, 10),
        tokensize=TOKENSIZE,
        ignore_bn=True,
        standardization=standardization,
    )
    lens.prepare(SHAPES, torch.device("cpu"))
    return lens


def make_models(batch, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return [{name: torch.randn(shape, generator=generator) for name, shape in SHAPES.items()} for _ in range(batch)]


def stack(models):
    return {name: torch.stack([model[name] for model in models]) for name in SHAPES}


def main():
    models = make_models(4)

    plain = make_lens({})
    tokens = plain._tokens(stack(models))
    for index, model in enumerate(models):
        expected, _mask, pos = tokenize(model, tokensize=TOKENSIZE, ignore_bn=True)
        assert torch.equal(tokens[index], expected), f"model {index}: tokens differ from diff_weight_pipe.tokenize"
        assert torch.equal(plain.pos, pos.to(torch.int)), f"model {index}: positions differ"
    print(f"OK  tokens are bit-for-bit equal to diff_weight_pipe.tokenize  {tuple(tokens.shape)}")

    standardized = make_lens(STATS)
    tokens = standardized._tokens(stack(models))
    for index, model in enumerate(models):
        expected, _mask, _pos = tokenize(standardize_state_dict(model, STATS), tokensize=TOKENSIZE, ignore_bn=True)
        assert torch.equal(tokens[index], expected), f"model {index}: standardized tokens differ"
    print("OK  standardized tokens are bit-for-bit equal")

    alone = torch.cat([standardized._tokens(stack([model])) for model in models])
    assert torch.equal(tokens, alone), "batched tokens differ from one-model-at-a-time tokens"
    print("OK  batch invariant: f(batch)[i] is bit-for-bit f(batch[i])")


if __name__ == "__main__":
    main()

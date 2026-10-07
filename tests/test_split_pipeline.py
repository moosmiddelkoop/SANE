"""End to end: a tiny zoo through consolidated preprocessing, with the fixed split."""

import json

import pytest
import torch

from SANE.datasets.dataset_preprocessing_consolidated import prepare_multiple_datasets
from SANE.datasets.zoo_split import MissingSplitError, check_dataset_splits, create_split, load_split
from SANE.git_re_basin.git_re_basin import zoo_cnn_permutation_spec

N_MODELS = 20
EPOCH = 1


def tiny_cnn_state(seed):
    """A checkpoint with the layer names of zoo_cnn_permutation_spec."""
    torch.manual_seed(seed)
    layers = {
        "module_list.0": torch.nn.Conv2d(1, 4, 3),
        "module_list.3": torch.nn.Conv2d(4, 4, 3),
        "module_list.6": torch.nn.Conv2d(4, 4, 3),
        "module_list.9": torch.nn.Linear(4, 8),
        "module_list.11": torch.nn.Linear(8, 10),
    }
    return {f"{name}.{k}": v for name, layer in layers.items() for k, v in layer.state_dict().items()}


@pytest.fixture
def zoo(tmp_path):
    zoo = tmp_path / "zoo"
    for i in range(N_MODELS):
        model = zoo / f"model_{i}"
        (model / f"checkpoint_{EPOCH:06d}").mkdir(parents=True)
        torch.save(tiny_cnn_state(i), model / f"checkpoint_{EPOCH:06d}" / "checkpoints")
        (model / "params.json").write_text(json.dumps({"seed": i}))
        lines = [{"training_iteration": e, "test_acc": 0.5 + i / 100} for e in range(EPOCH + 1)]
        (model / "result.json").write_text("\n".join(json.dumps(line) for line in lines))
    return zoo


def config(zoo, target, **overrides):
    return {
        "dataset_target_path": target,
        "zoo_path": [zoo],
        "epoch_list": [EPOCH],
        "permutation_spec": zoo_cnn_permutation_spec,
        "map_to_canonical": False,
        "standardize": True,
        "splits": ["train", "val", "test"],
        "max_samples": None,
        "weight_threshold": float("inf"),
        "property_keys": {"result_keys": ["test_acc"], "config_keys": []},
        "num_threads": 2,
        "windowsize": 8,
        "supersample": 1,
        "precision": "32",
        "ignore_bn": True,
        "tokensize": 0,
        **overrides,
    }


def test_preprocessing_needs_a_split(zoo, tmp_path):
    with pytest.raises(MissingSplitError):
        prepare_multiple_datasets([config(zoo, tmp_path / "out")])


def test_preprocessing_rejects_old_split_flags(zoo, tmp_path):
    create_split(zoo)
    with pytest.raises(ValueError, match="no longer used"):
        prepare_multiple_datasets([config(zoo, tmp_path / "out", shuffle_path=True)])


def test_preprocessing_follows_the_split(zoo, tmp_path):
    split = create_split(zoo)
    target = tmp_path / "out"
    prepare_multiple_datasets([config(zoo, target)])

    datasets = torch.load(target / "dataset.pt", weights_only=False)
    assert check_dataset_splits(datasets, "test") == split.split_id
    for name in ("train", "val", "test"):
        ds = datasets[f"{name}set"]
        assert ds.models == [str(p) for p in load_split(zoo).paths(name)]
        assert len(ds) == len(split.models[name])
        info = json.loads((target / f"dataset_info_{name}.json").read_text())
        assert info["split_id"] == split.split_id
        assert info["models"] == ds.models

import json
from types import SimpleNamespace

import pytest

from SANE.datasets.zoo_split import (
    SPLIT_FILE,
    MissingSplitError,
    SplitMismatchError,
    assert_same_split,
    check_dataset_splits,
    create_split,
    load_split,
    resplit_train_val,
    select_models,
)


@pytest.fixture
def zoo(tmp_path):
    zoo = tmp_path / "zoo"
    for i in range(20):
        (zoo / f"model_{i}").mkdir(parents=True)
    return zoo


def test_create_split_partitions_all_models(zoo):
    split = create_split(zoo)
    names = split.models["train"] + split.models["val"] + split.models["test"]
    assert sorted(names) == sorted(p.name for p in zoo.iterdir() if p.is_dir())
    assert len(set(names)) == len(names)
    assert [len(split.models[s]) for s in ("train", "val", "test")] == [14, 3, 3]
    assert load_split(zoo) == split


def test_create_split_refuses_to_overwrite(zoo):
    create_split(zoo)
    before = (zoo / SPLIT_FILE).read_text()
    with pytest.raises(FileExistsError):
        create_split(zoo, seed=1)
    assert (zoo / SPLIT_FILE).read_text() == before


def test_split_does_not_depend_on_directory_order(tmp_path):
    zoos = []
    for name, order in [("a", range(20)), ("b", reversed(range(20)))]:
        zoo = tmp_path / name
        for i in order:
            (zoo / f"model_{i}").mkdir(parents=True)
        zoos.append(create_split(zoo))
    assert zoos[0].models == zoos[1].models
    assert zoos[0].split_id == zoos[1].split_id


def test_missing_split_fails_with_instructions(zoo):
    with pytest.raises(MissingSplitError, match="create_split.py"):
        load_split(zoo)
    with pytest.raises(MissingSplitError):
        select_models([zoo], "train")


def test_edited_split_is_rejected(zoo):
    create_split(zoo)
    record = json.loads((zoo / SPLIT_FILE).read_text())
    record["train"].append(record["test"].pop())
    (zoo / SPLIT_FILE).write_text(json.dumps(record))
    with pytest.raises(SplitMismatchError, match="edited"):
        load_split(zoo)


def test_changed_zoo_is_rejected(zoo):
    create_split(zoo)
    (zoo / "model_new").mkdir()
    with pytest.raises(SplitMismatchError, match="changed"):
        load_split(zoo)


def test_select_models_shares_reference_and_caps_proportionally(zoo):
    create_split(zoo)
    train, ref_train, split_id = select_models([zoo], "train")
    test, ref_test, _ = select_models([zoo], "test")
    assert not set(train) & set(test)
    assert ref_train == ref_test == train
    capped, _, _ = select_models([zoo], "train", max_samples=10)
    assert capped == train[:7]
    assert split_id == load_split(zoo).split_id


def test_select_all_needs_no_split(zoo):
    paths, _, split_id = select_models([zoo], "all")
    assert len(paths) == 20 and split_id is None


def test_check_dataset_splits(zoo):
    split = create_split(zoo)

    def ds(name, split_id=split.split_id):
        return SimpleNamespace(split_id=split_id, models=[str(p) for p in split.paths(name)])

    datasets = {"trainset": ds("train"), "valset": ds("val"), "testset": ds("test")}
    assert check_dataset_splits(datasets, "x") == split.split_id

    with pytest.raises(SplitMismatchError, match="different splits"):
        check_dataset_splits({**datasets, "testset": ds("test", "other")}, "x")
    with pytest.raises(SplitMismatchError, match="both"):
        check_dataset_splits({**datasets, "testset": ds("train")}, "x")
    with pytest.raises(SplitMismatchError, match="no split_id"):
        check_dataset_splits({**datasets, "testset": SimpleNamespace()}, "x")


def test_assert_same_split(tmp_path, zoo):
    split = create_split(zoo)
    dump = tmp_path / "dataset.pt"
    info = tmp_path / "dataset_info_train.json"
    config = {"dataset::dump": str(dump)}
    dataset = SimpleNamespace(split_id=split.split_id)

    info.write_text(json.dumps({"split_id": split.split_id}))
    assert_same_split(config, dataset)

    info.write_text(json.dumps({"split_id": "other"}))
    with pytest.raises(SplitMismatchError, match="pretrained on split other"):
        assert_same_split(config, dataset)

    info.write_text(json.dumps({}))
    with pytest.raises(SplitMismatchError, match="no split_id"):
        assert_same_split(config, dataset)
    assert_same_split({**config, "dataset::legacy_unverified_split": True}, dataset)


def test_resplit_train_val_keeps_test_out(zoo):
    split = create_split(zoo)
    train, val, test = (split.models[s] for s in ("train", "val", "test"))
    seen_val = set()
    for seed in range(10):
        new_train, new_val = resplit_train_val(train, val, seed)
        assert sorted(new_train + new_val) == sorted(train + val)
        assert len(new_val) == len(val)
        assert not set(new_train + new_val) & set(test)
        seen_val |= set(new_val)
    assert seen_val - set(val), "resplitting should move models between train and val"
    assert resplit_train_val(train, val, 3) == resplit_train_val(train, val, 3)

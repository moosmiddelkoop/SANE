"""The fixed train/val/test split of a model zoo.

Every zoo has exactly one split, stored as ``<zoo>/split.json``. It is created
once, before any other work on the zoo, with::

    uv run data/create_split.py <zoo>

and never changes afterwards. All dataset classes read the models of a split
from this file, so preprocessing, pretraining and downstream tasks cannot end
up on different splits. There is deliberately no shuffle or ratio option at
load time.

Each split has a ``split_id`` (a hash of its model lists). Preprocessing stores
it with the dataset, pretraining checks that train/val/test share it, and
downstream scripts check that the zoo's current split still matches the one
the encoder was pretrained on (``assert_same_split``).
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Union

SPLIT_FILE = "split.json"
SPLITS = ("train", "val", "test")
ALL = "all"  # every model in the root, for populations that are not a zoo


class MissingSplitError(FileNotFoundError):
    """The zoo has no split.json yet."""


class SplitMismatchError(RuntimeError):
    """Data that should share one split does not."""


@dataclass(frozen=True)
class ZooSplit:
    zoo: Path
    models: Dict[str, List[str]]  # split name -> model directory names
    split_id: str

    def paths(self, split: str) -> List[Path]:
        return [self.zoo / name for name in self.models[split]]


def model_dirs(zoo: Union[str, Path]) -> List[str]:
    """Names of all model directories in a zoo, sorted.

    ``iterdir()`` order depends on the filesystem, so it is never used as is.
    """
    return sorted(d.name for d in Path(zoo).iterdir() if d.is_dir())


def compute_split_id(models: Dict[str, List[str]]) -> str:
    payload = json.dumps({s: models[s] for s in SPLITS}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def create_split(
    zoo: Union[str, Path],
    ratios: Sequence[float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> ZooSplit:
    """Create the split of a zoo and write it to ``<zoo>/split.json``.

    Refuses to run if the zoo already has a split: a split is made once. To
    redo it, delete the file by hand, knowing that every dataset and encoder
    built on the old split becomes unusable.
    """
    zoo = Path(zoo).absolute()
    split_path = zoo / SPLIT_FILE
    if split_path.exists():
        raise FileExistsError(
            f"{split_path} already exists. A zoo's split is created once and never "
            "changed. Delete the file by hand only if you really want a new split; "
            "every dataset.pt and encoder built on the old one will stop loading."
        )
    if len(ratios) != len(SPLITS) or abs(sum(ratios) - 1.0) > 1e-8 or min(ratios) < 0:
        raise ValueError(f"ratios must be 3 non-negative numbers summing to 1, got {ratios}")

    names = model_dirs(zoo)
    if not names:
        raise ValueError(f"no model directories found in {zoo}")
    random.Random(seed).shuffle(names)
    n_train = int(ratios[0] * len(names))
    n_val = int(ratios[1] * len(names))
    models = {
        "train": names[:n_train],
        "val": names[n_train : n_train + n_val],
        "test": names[n_train + n_val :],
    }
    split_id = compute_split_id(models)
    record = {
        "split_id": split_id,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ratios": list(ratios),
        "seed": seed,
        "n_models": len(names),
        **models,
    }
    # mode "x" fails if another process created the file in the meantime
    with split_path.open("x") as f:
        json.dump(record, f, indent=2)
    logging.info(
        f"created split {split_id} for {zoo}: "
        + ", ".join(f"{s}={len(models[s])}" for s in SPLITS)
    )
    return ZooSplit(zoo=zoo, models=models, split_id=split_id)


def load_split(zoo: Union[str, Path]) -> ZooSplit:
    """Load and validate the split of a zoo. Fails if there is none."""
    zoo = Path(zoo).absolute()
    split_path = zoo / SPLIT_FILE
    if not split_path.exists():
        raise MissingSplitError(
            f"No {SPLIT_FILE} in {zoo}. Every zoo needs a fixed train/val/test split "
            "before any preprocessing, pretraining or downstream task. Create it once "
            f"with:\n\n    uv run data/create_split.py {zoo}\n"
        )
    record = json.loads(split_path.read_text())
    models = {s: list(record[s]) for s in SPLITS}

    if compute_split_id(models) != record["split_id"]:
        raise SplitMismatchError(
            f"{split_path} was edited: its model lists no longer match its split_id."
        )
    seen = {}
    for s in SPLITS:
        for name in models[s]:
            if name in seen:
                raise SplitMismatchError(f"{split_path}: model {name} is in both {seen[name]} and {s}")
            seen[name] = s
    on_disk = set(model_dirs(zoo))
    missing, unlisted = set(seen) - on_disk, on_disk - set(seen)
    if missing or unlisted:
        raise SplitMismatchError(
            f"The zoo {zoo} changed since its split was created: "
            f"{len(missing)} listed models are gone, {len(unlisted)} model directories "
            f"are not in {SPLIT_FILE} (e.g. {sorted(missing | unlisted)[:3]})."
        )
    return ZooSplit(zoo=zoo, models=models, split_id=record["split_id"])


def select_models(roots: Sequence[Path], split: str, max_samples=None):
    """Model paths of one split, over one or more zoos.

    Returns ``(paths, reference_candidates, split_id)``. The reference
    candidates are the train models, so every split of a zoo maps to the same
    canonical reference model. ``max_samples`` caps the total number of models
    of the zoo; each split keeps its proportional share, taken from the front
    of its (already shuffled) list.

    ``split="all"`` returns every model and needs no split file. It is meant for
    sampled populations, never for building train/val/test data.
    """
    if split == ALL:
        paths = [Path(r) / name for r in roots for name in model_dirs(r)]
        if max_samples is not None:
            paths = paths[:max_samples]
        return paths, paths, None
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS} or '{ALL}', got {split!r}")

    paths, reference_candidates, split_ids = [], [], []
    for root in roots:
        zoo_split = load_split(root)
        selected = zoo_split.paths(split)
        if max_samples is not None:
            n_total = sum(len(v) for v in zoo_split.models.values())
            selected = selected[: int(max_samples * len(selected) / n_total)]
        paths.extend(selected)
        reference_candidates.extend(zoo_split.paths("train"))
        split_ids.append(zoo_split.split_id)
    return paths, reference_candidates, "+".join(split_ids)


def resplit_train_val(train: Sequence, val: Sequence, seed: int):
    """Shuffle which models are train and which are val, for one seed.

    For cross-validation and split-variance experiments. Only train and val
    are pooled and re-cut; the test models are never passed in, so the fixed
    test split stays held out. Val keeps its size from split.json.
    Returns ``(train, val)``.
    """
    pool = list(train) + list(val)
    random.Random(seed).shuffle(pool)
    return pool[len(val) :], pool[: len(val)]


def check_dataset_splits(datasets: Dict[str, object], what: str):
    """Check that the train/val/test datasets of one dataset.pt share one split.

    Each dataset carries the ``split_id`` and ``models`` it was built from
    (set at preprocessing). Returns the shared split_id.
    """
    present = {k: d for k, d in datasets.items() if d is not None}
    split_ids = {k: getattr(d, "split_id", None) for k, d in present.items()}
    missing = [k for k, split_id in split_ids.items() if split_id is None]
    if missing:
        raise SplitMismatchError(
            f"{what}: {missing} carry no split_id, so they were preprocessed before "
            "zoos had a fixed split.json and their split cannot be verified. Re-run "
            "the preprocessing, or set config['dataset::legacy_unverified_split'] = True "
            "to train on them anyway."
        )
    if len(set(split_ids.values())) != 1:
        raise SplitMismatchError(f"{what}: datasets come from different splits {split_ids}")
    seen = {}
    for k, d in present.items():
        for name in getattr(d, "models"):
            if name in seen and seen[name] != k:
                raise SplitMismatchError(f"{what}: model {name} is in both {seen[name]} and {k}")
            seen[name] = k
    return next(iter(split_ids.values()))


def assert_same_split(pretrain_config: dict, dataset) -> None:
    """Check that a downstream dataset uses the split the encoder was pretrained on.

    ``pretrain_config`` is the encoder's params.json. Its ``dataset::dump`` points
    to the dataset.pt; the dataset_info_train.json next to it holds the split_id.
    """
    info_path = Path(pretrain_config["dataset::dump"]).parent / "dataset_info_train.json"
    pretrain_split_id = json.loads(info_path.read_text()).get("split_id")
    if pretrain_split_id is None:
        if pretrain_config.get("dataset::legacy_unverified_split", False):
            logging.warning(
                "dataset::legacy_unverified_split is set: the encoder's pretraining split "
                "is unknown, so its train models may overlap with this downstream test set"
            )
            return
        raise SplitMismatchError(
            f"{info_path} has no split_id: the encoder was pretrained on data made "
            "before zoos had a fixed split.json, so its train models may overlap "
            "with this downstream test set. Set "
            "config['dataset::legacy_unverified_split'] = True to run anyway."
        )
    if pretrain_split_id != dataset.split_id:
        raise SplitMismatchError(
            f"The encoder was pretrained on split {pretrain_split_id}, but this "
            f"dataset uses split {dataset.split_id}."
        )

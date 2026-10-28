# prepare data: every 5th checkpoint (epochs 0, 5, ..., 60) of the full ResNet-18 CIFAR-100 zoo
# in one dataset. Same settings as preprocess_dataset_resnet18_cifar100_tk288.py (epoch 60 only),
# except the zoo path, the epoch list, the supersample and n_chunks.
import logging
import os

# Single-threaded BLAS per worker: preprocessing parallelizes across CPUs via Ray,
# so multi-threaded BLAS would oversubscribe cores. Must be set before torch import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

from SANE.datasets.dataset_preprocessing_consolidated import prepare_multiple_datasets
from SANE.git_re_basin.git_re_basin import (
    resnet18_permutation_spec,
)

logging.basicConfig(level=logging.INFO)


def prep_data():
    dataset_target_path = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/dataset_cifar100_token_288_ep0-60_std/")
    # same 1000 models and the same split.json (split 35e0c40d4d11) as the epoch-60 zoo
    zoo_path = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform/")
    configurations = create_configurations([(zoo_path, resnet18_permutation_spec, dataset_target_path)], filter_fn=None)
    prepare_multiple_datasets(configurations=configurations)


def create_configurations(zoo_path_and_permutation_spec_and_target_path, filter_fn=None):
    # static parameters
    epoch_list = list(range(0, 61, 5))  # 13 checkpoints per model
    map_to_canonical = True
    standardize = True
    max_samples = None
    weight_threshold = float("inf")
    num_threads = int(os.environ["SLURM_CPUS_PER_TASK"])  # CPUs requested for this SLURM job (also correct under --exclusive)
    windowsize = 1024
    # stored windows per checkpoint. The epoch-60 dataset stores 38 windows per model; 3 windows x 13
    # checkpoints = 39 per model keeps dataset.pt at the same size (~150 GB), so pretraining fits in RAM
    # as before. Each checkpoint then covers about 3/38 of its 39304 tokens.
    supersample = 3
    precision = "16"  # tokens cast to fp16 at stacking; halves dataset.pt
    ignore_bn = True
    tokensize = 288
    # all raw checkpoints of train are 700 models x 13 x 45 MB ~ 410 GB, more than a genoa node (336 GB).
    # Ray canonicalization holds a chunk about 3-4x (object store, workers, results), so 7 chunks
    # (~60 GB raw each) went out of memory. 28 chunks of 25 train models keep ~15 GB raw at a time.
    n_chunks = 28

    # permutation spec
    # each stored sample holds permutation_number + 1 copies of its window
    permutation_number_train = 5
    permutations_per_sample_train = 5
    permutation_number_test = 5
    permutations_per_sample_test = 5

    splits = ["train", "val", "test"]
    page_size = 2**27

    result_key_list = ["test_acc", "training_iteration", "ggap"]
    config_key_list = ["model::type"]
    property_keys = {
        "result_keys": result_key_list,
        "config_keys": config_key_list,
    }

    configurations = []
    for split in splits:
        # dynamic parameters
        for zoo_path, permutation_spec, dataset_target_path in zoo_path_and_permutation_spec_and_target_path:
            configurations.append(
                {
                    "dataset_target_path": dataset_target_path,
                    "zoo_path": zoo_path,
                    "epoch_list": epoch_list,
                    "permutation_spec": permutation_spec,
                    "map_to_canonical": map_to_canonical,
                    "standardize": standardize,
                    "max_samples": max_samples,
                    "weight_threshold": weight_threshold,
                    "num_threads": num_threads,
                    "windowsize": windowsize,
                    "supersample": supersample,
                    "precision": precision,
                    "ignore_bn": ignore_bn,
                    "tokensize": tokensize,
                    "permutation_number_train": permutation_number_train,
                    "permutations_per_sample_train": permutations_per_sample_train,
                    "permutation_number_test": permutation_number_test,
                    "permutations_per_sample_test": permutations_per_sample_test,
                    "page_size": page_size,
                    "splits": [split],
                    "property_keys": property_keys,
                    "drop_pt_dataset": False,
                    "filter_fn": filter_fn,
                    "n_chunks": n_chunks,
                }
            )

    return configurations


if __name__ == "__main__":
    prep_data()

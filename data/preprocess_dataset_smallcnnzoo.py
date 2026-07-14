# prepare data
import logging
import os

from argparse import ArgumentParser

# Single-threaded BLAS per worker: preprocessing parallelizes across CPUs via Ray
# (many lightweight checkpoint-loading workers), so multi-threaded BLAS would
# oversubscribe cores without helping the tiny per-model tensor ops. Must be set
# before torch/numpy import.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

from SANE.datasets.dataset_preprocessing_consolidated import prepare_multiple_datasets
from SANE.git_re_basin.git_re_basin import smallcnnzoo_permutation_spec

logging.basicConfig(level=logging.INFO)

OUT_DIR = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist_preprocessed/")
IN_DIR = Path("/projects/prjs2156/shared/wsl/unthi_zoo/unthi_mnist/")

def prep_data(out_dir=OUT_DIR, in_dir=IN_DIR):
    dataset_target_path = [out_dir,]
    # mkdir target path if it does not exist
    for path in dataset_target_path:
        path.mkdir(parents=True, exist_ok=True)

    zoo_path = [in_dir]
    zoo_path_and_permutation_spec_and_target_path = [
        (zoo_path[0], smallcnnzoo_permutation_spec, dataset_target_path[0]),
    ]
    configurations = create_configurations(zoo_path_and_permutation_spec_and_target_path, filter_fn=None)
    prepare_multiple_datasets(configurations=configurations)


def create_configurations(zoo_path_and_permutation_spec_and_target_path, filter_fn=None):
    # static parameters
    epoch_list = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    map_to_canonical = True
    standardize = True
    ds_split = [0.7, 0.15, 0.15]
    max_samples = None  # for smoke tests (truncates the amount of models being preprocessed)
    weight_threshold = 100  # drops any checkoint with blown up weights of a magnitude above this threshold
    # use all CPU cores allocated to this job (respects the SLURM/cgroup allocation
    # on Snellius); falls back to the full machine count when off-cluster
    # try:
    #     num_threads = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]  # Linux-only
    # except AttributeError:
    #     num_threads = os.cpu_count()
    num_threads = 12
    shuffle_path = True
    windowsize = 58
    supersample = 1
    precision = "32"
    ignore_bn = True
    tokensize = 145

    # permutation spec
    permutation_number_train = 5
    permutations_per_sample_train = 5 # dead parameter, only used in PermutationAugmentation, which is not used in the code
    permutation_number_test = 5
    permutations_per_sample_test = 5 # dead parameter, only used in PermutationAugmentation, which is not used in the code

    # dataset splits
    splits = ["train", "val", "test"]

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
                    "ds_split": ds_split,
                    "max_samples": max_samples,
                    "weight_threshold": weight_threshold,
                    "num_threads": num_threads,
                    "shuffle_path": shuffle_path,
                    "windowsize": windowsize,
                    "supersample": supersample,
                    "precision": precision,
                    "ignore_bn": ignore_bn,
                    "tokensize": tokensize,
                    "permutation_number_train": permutation_number_train,
                    "permutations_per_sample_train": permutations_per_sample_train,
                    "permutation_number_test": permutation_number_test,
                    "permutations_per_sample_test": permutations_per_sample_test,
                    "splits": [split],
                    "property_keys": property_keys,
                    "drop_pt_dataset": False,
                    "filter_fn": filter_fn,
                }
            )

    return configurations


if __name__ == "__main__":
    ARGS = ArgumentParser()
    ARGS.add_argument("--out_dir", type=Path, default=OUT_DIR, help="Output directory for preprocessed dataset")
    ARGS.add_argument("--in_dir", type=Path, default=IN_DIR, help="Input directory containing the original dataset")
    args = ARGS.parse_args()
    prep_data(out_dir=args.out_dir, in_dir=args.in_dir)

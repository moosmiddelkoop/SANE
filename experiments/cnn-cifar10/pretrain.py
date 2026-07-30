"""
Plain-PyTorch SANE pretraining — Ray-free replacement for pretrain_sane_cifar10_cnn.py.

Reuses AEModule and get_transformations; mirrors AE_trainable.step_ssl semantics so
metrics are comparable with the Ray runs. Outputs are drop-in compatible with the
downstream scripts: params.json + checkpoint_XXXXXX/state.pt in the output dir.

Run from this directory: `uv run python pretrain.py` (add --smoke-test for a quick
1-epoch CPU check without torch.compile).
"""
import logging

logging.basicConfig(level=logging.INFO)

import os

# Snellius A100 node: 18 CPU cores per GPU. With 8 DataLoader workers + 1 main
# process, 2 BLAS threads each saturates the allocation without oversubscription.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from SANE.datasets.augmentations import MultiWindowCutter
from SANE.models.def_AE_module import AEModule
from SANE.models.def_AE_trainable import get_transformations

PATH_ROOT = Path("./")


def get_config():
    # same hyperparameters as pretrain_sane_cifar10_cnn.py, minus the Ray-only keys
    config = {}
    config["seed"] = 32
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["training::precision"] = "amp"
    config["trainset::batchsize"] = 32

    config["ae:transformer_type"] = "gpt2"
    config["model::compile"] = True

    # permutation specs
    config["training::permutation_number"] = 5
    config["training::view_2_canon"] = True
    config["testing::permutation_number"] = 5
    config["testing::view_1_canon"] = True
    config["testing::view_2_canon"] = False

    config["ae:i_dim"] = 289
    config["ae:lat_dim"] = 128
    config["ae:max_positions"] = [100, 10, 40]
    config["training::windowsize"] = 64
    config["ae:d_model"] = 1024
    config["ae:nhead"] = 8
    config["ae:num_layers"] = 8

    # configure optimizer
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = 1e-4
    config["optim::wd"] = 3e-9
    config["optim::scheduler"] = "OneCycleLR"

    # training config
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.05
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr"
    config["training::epochs_train"] = 50
    config["training::output_epoch"] = 5
    config["training::test_epochs"] = 1

    # dataset
    data_path = Path("../../data/dataset_cnn_cifar10_sample_ep21-25_std/")
    config["dataset::dump"] = data_path.joinpath("dataset.pt").absolute()
    config["downstreamtask::dataset"] = None

    # augmentations
    config["trainloader::workers"] = 8
    config["trainset::add_noise_view_1"] = 0.1
    config["trainset::add_noise_view_2"] = 0.1
    config["trainset::noise_multiplicative"] = True
    config["trainset::erase_augment_view_1"] = None
    config["trainset::erase_augment_view_2"] = None

    return config


def build_dataloaders(config):
    # ported from AE_trainable.load_datasets
    windowsize = config.get("training::windowsize", 15)
    trafo_dataset = None
    if config.get("trainset::multi_windows", None):
        trafo_dataset = MultiWindowCutter(
            windowsize=windowsize, k=config.get("trainset::multi_windows")
        )

    logging.info("Load Data")
    # dataset.pt pickles PreprocessedSamplingDataset objects (our own, trusted),
    # which torch>=2.6 refuses to load with the new weights_only=True default
    dataset = torch.load(config["dataset::dump"], weights_only=False)
    trainset = dataset["trainset"]
    testset = dataset["testset"]
    valset = dataset.get("valset", None)

    if trafo_dataset is not None:
        trainset.transforms = trafo_dataset
        testset.transforms = trafo_dataset
        if valset is not None:
            valset.transforms = trafo_dataset

    logging.info("set up dataloaders")
    # correct dataloader batchsize with # of multi_window samples out of single __getitem__ call
    assert (
        config["trainset::batchsize"] % config.get("trainset::multi_windows", 1) == 0
    ), f'batchsize {config["trainset::batchsize"]} needs to be divisible by multi_windows {config["trainset::multi_windows"]}'
    bs_corr = int(
        config["trainset::batchsize"] / config.get("trainset::multi_windows", 1)
    )
    logging.info(f"corrected batchsize to {bs_corr}")

    train_workers = config.get("trainloader::workers", 2)
    test_workers = config.get("testloader::workers", 2)
    trainloader = DataLoader(
        trainset,
        batch_size=bs_corr,
        shuffle=True,
        drop_last=True,  # important: we need equal batch sizes
        num_workers=train_workers,
        prefetch_factor=4 if train_workers > 0 else None,
    )
    testloader = DataLoader(
        testset,
        batch_size=bs_corr,
        shuffle=False,
        drop_last=True,  # important: we need equal batch sizes
        num_workers=test_workers,
        prefetch_factor=4 if test_workers > 0 else None,
    )
    valloader = None
    if valset is not None:
        valloader = DataLoader(
            valset,
            batch_size=bs_corr,
            shuffle=False,
            drop_last=True,  # important: we need equal batch sizes
            num_workers=test_workers,
            prefetch_factor=4 if test_workers > 0 else None,
        )

    return trainloader, testloader, valloader


def evaluate(module, trainloader, testloader, valloader, epoch):
    # eval-only pass over all splits; mirrors the "test first validation mode"
    result = {}
    perf = module.test_epoch(trainloader, epoch=epoch)
    result.update({f"{key}_train": value for key, value in perf.items()})
    perf = module.test_epoch(testloader, epoch=epoch)
    result.update({f"{key}_test": value for key, value in perf.items()})
    if valloader is not None:
        perf = module.test_epoch(valloader, epoch=epoch)
        result.update({f"{key}_val": value for key, value in perf.items()})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="1 epoch, no torch.compile, no dataloader workers — quick correctness check",
    )
    args = parser.parse_args()

    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")

    config = get_config()
    experiment_name = "sane_cifar10_cnn_plain"
    if args.smoke_test:
        experiment_name = "sane_cifar10_cnn_smoke_test"
        config["training::epochs_train"] = 1
        config["training::output_epoch"] = 1
        config["model::compile"] = False
        config["trainloader::workers"] = 0
        config["testloader::workers"] = 0

    output_dir = PATH_ROOT.joinpath("sane_pretraining", experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainloader, testloader, valloader = build_dataloaders(config)
    config["training::steps_per_epoch"] = len(trainloader)

    logging.info("instanciate model")
    module = AEModule(config=config)
    trafo_train, trafo_test, trafo_dst = get_transformations(config)
    module.set_transforms(trafo_train, trafo_test, trafo_dst)

    # params.json plays the same role as Ray's: downstream scripts read the config from it
    with output_dir.joinpath("params.json").open("w") as f:
        json.dump(config, f, indent=4, default=str)

    # resume from latest checkpoint if one exists in the output dir
    start_epoch = 1
    checkpoints = sorted(output_dir.glob("checkpoint_*"))
    if checkpoints:
        latest = checkpoints[-1]
        logging.info(f"resume from {latest}")
        module.load_model(latest)
        start_epoch = int(latest.name.split("_")[-1]) + 1

    results_path = output_dir.joinpath("results.json")
    results = []
    if results_path.exists():
        results = json.load(results_path.open("r"))

    def log_result(result):
        results.append(result)
        with results_path.open("w") as f:
            json.dump(results, f, indent=4)
        metrics = {k: v for k, v in result.items() if "loss/loss" in k}
        print(f"epoch {result['training_iteration']}: {metrics}")

    epochs_train = config["training::epochs_train"]
    output_epoch = config["training::output_epoch"]

    if start_epoch == 1:
        print("test first validation mode")
        result = evaluate(module, trainloader, testloader, valloader, epoch=0)
        result["training_iteration"] = 0
        log_result(result)

    for epoch in range(start_epoch, epochs_train + 1):
        result = {"training_iteration": epoch}
        # TRAIN EPOCH(s): run several training epochs before one test epoch
        perf = {}
        for _ in range(config["training::test_epochs"]):
            perf = module.train_epoch(trainloader, epoch=epoch, show_progress=True)
        result.update({f"{key}_train": value for key, value in perf.items()})
        # TEST / VALIDATION EPOCH
        perf = module.test_epoch(testloader, epoch=epoch)
        result.update({f"{key}_test": value for key, value in perf.items()})
        if valloader is not None:
            perf = module.test_epoch(valloader, epoch=epoch)
            result.update({f"{key}_val": value for key, value in perf.items()})
        log_result(result)

        if epoch % output_epoch == 0 or epoch == epochs_train:
            checkpoint_dir = output_dir.joinpath(f"checkpoint_{epoch:06d}")
            module.save_model(checkpoint_dir)
            logging.info(f"saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    main()

import logging

logging.basicConfig(level=logging.INFO)

import os

# set environment variables to limit cpu usage
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import ray
import torch
from ray.air.integrations.wandb import WandbLoggerCallback

from SANE.models.def_AE_trainable import AE_trainable
from SANE.utils import seed_everything

# one run per tokensize (288 or 576); each has its own preprocessed dataset
parser = ArgumentParser()
parser.add_argument("--tokensize", type=int, required=True)
parser.add_argument("--epochs", type=int, default=50)
TOKENSIZE = parser.parse_args().tokensize
EPOCHS = parser.parse_args().epochs

OUTPUT_PATH = Path("/projects/prjs2156/shared/wsl/metanets/sane_pretraining")
WANDB_PROJECT = "sane-pretraining-resnet18"
# directory within OUTPUT_PATH where the results will be stored
EXPERIMENT_NAME = "cifar100_resnet18"
# Names the trial dir within EXPERIMENT_NAME and the W&B run
# (the trial_id suffix keeps names unique across launches)
RUN_TAG = f"cifar100-resnet18-tk{TOKENSIZE}-v1.0"

DATA_PATH = Path(
    os.environ.get(
        "SANE_DATA_DIR",
        f"/projects/prjs2156/shared/wsl/cifar100_resnet18/dataset_cifar100_token_{TOKENSIZE}_ep60_std/",
    )
)


def main():
    ### set experiment resources ####
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # ray init to limit memory and storage
    cpus_per_trial = 16  # Snellius H100 node: 16 CPU cores per GPU
    gpus_per_trial = 1
    gpus = 1
    cpus = gpus * cpus_per_trial

    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    print(f"resources_per_trial: {resources_per_trial}")

    ### configure experiment #########
    experiment_name = EXPERIMENT_NAME
    # set module parameters
    config = {}
    config["seed"] = 32
    seed_everything(config["seed"])
    config["device"] = "cuda"
    config["device_no"] = 1
    config["training::precision"] = "amp"
    # 1024 x 32 is the largest window that keeps batchsize 32 on one H100 (85 GiB peak,
    # see probe_windowsize_memory.py)
    config["trainset::batchsize"] = 32

    config["ae:transformer_type"] = "gpt2"
    config["model::compile"] = True

    # permutation specs
    # view 2 is a random stored permutation (upstream set view_2_canon = True, which makes
    # both views canonical and leaves the stored permutations unused)
    config["training::permutation_number"] = (
        5  # any nonzero value behaves identically, it is only checked to be == 0 or not
    )
    config["training::view_1_canon"] = True
    config["training::view_2_canon"] = False
    config["testing::permutation_number"] = 5
    config["testing::view_1_canon"] = True
    config["testing::view_2_canon"] = False

    config["training::reduction"] = "mean"

    config["ae:i_dim"] = TOKENSIZE
    config["ae:lat_dim"] = 128
    # must be bigger than [tokens per model, layers, max channels]: data has [39304 (tk288) / 20132 (tk576), 21, 512]
    config["ae:max_positions"] = [55000, 100, 550]
    config["training::windowsize"] = 1024
    config["ae:d_model"] = 2048
    config["ae:nhead"] = 16
    config["ae:num_layers"] = 8

    # configure optimizer
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = 1e-4
    config["optim::wd"] = 3e-9
    config["optim::scheduler"] = "OneCycleLR"
    # clip gradients (no clipping near the OneCycleLR peak caused the c898d loss spike)
    config["training::gradient_clipping"] = "norm"
    config["training::gradient_clipp_value"] = 2.0

    # training config
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.05
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr"
    # AMP
    #
    config["training::epochs_train"] = EPOCHS
    config["training::output_epoch"] = 5
    config["training::test_epochs"] = 1
    # development phase: monitor on the val split only, hold out the test split for final evals
    config["training::eval_testset"] = False

    config["monitor_memory"] = True

    # configure output path
    output_dir = OUTPUT_PATH
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass

    ###### Datasets ###########################################################################
    # fail now on a wrong path (a mkdir here used to create an empty dir and fail later)
    assert DATA_PATH.joinpath("dataset.pt").exists(), f"no dataset.pt in {DATA_PATH}"
    config["dataset::dump"] = DATA_PATH.joinpath("dataset.pt").absolute()
    config["downstreamtask::dataset"] = None

    ### Augmentations
    config["trainloader::workers"] = 8
    config["trainset::add_noise_view_1"] = 0.1
    config["trainset::add_noise_view_2"] = 0.1
    config["trainset::noise_multiplicative"] = True  # dead key
    config["trainset::erase_augment_view_1"] = None
    config["trainset::erase_augment_view_2"] = None

    config["callbacks"] = []

    config["resources"] = resources_per_trial
    context = ray.init(
        num_cpus=cpus,
        num_gpus=gpus,
        include_dashboard=False,  # monitoring is via W&B; avoids port 8265 collisions on shared nodes
    )
    assert ray.is_initialized() == True

    print("started ray.")

    experiment = ray.tune.Experiment(
        name=experiment_name,
        run=AE_trainable,
        stop={
            "training_iteration": config["training::epochs_train"],
        },
        checkpoint_config=ray.air.CheckpointConfig(
            num_to_keep=None,
            checkpoint_frequency=config["training::output_epoch"],
            checkpoint_at_end=True,
        ),
        config=config,
        local_dir=output_dir,
        resources_per_trial=resources_per_trial,
        trial_name_creator=lambda trial: f"{RUN_TAG}_{trial.trial_id}",
        # date suffix (same format as ray's default dirname) makes chronological ordering explicit
        trial_dirname_creator=lambda trial: (
            f"{RUN_TAG}_{trial.trial_id}_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        ),
    )
    # run
    ray.tune.run_experiments(
        experiments=experiment,
        resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        reuse_actors=False,
        verbose=3,
        callbacks=[WandbLoggerCallback(project=WANDB_PROJECT)],
    )

    ray.shutdown()
    assert ray.is_initialized() == False


if __name__ == "__main__":
    main()

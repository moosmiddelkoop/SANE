import logging

logging.basicConfig(level=logging.INFO)

import os

# Snellius A100 node: 18 CPU cores per GPU. With 8 DataLoader workers + 1 main
# process, 2 BLAS threads each saturates the allocation without oversubscription.
# claude added this
# os.environ["OMP_NUM_THREADS"] = "2"
# os.environ["OPENBLAS_NUM_THREADS"] = "2"
# os.environ["MKL_NUM_THREADS"] = "2"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
# os.environ["NUMEXPR_NUM_THREADS"] = "2"

from datetime import datetime
from pathlib import Path

import ray
import torch
from ray.air.integrations.wandb import WandbLoggerCallback

from SANE.models.def_AE_trainable import AE_trainable

OUTPUT_PATH = Path("/projects/prjs2156/shared/wsl/metanets/sane_pretraining")
# short, informative human-readable tag for this launch: names the trial dir and the W&B run
# (the trial_id suffix keeps names unique across launches)
RUN_TAG = "pretraining-cifar10-v1.0"
DATA_PATH = Path(os.environ.get("SANE_DATA_DIR", "/projects/prjs2156/shared/wsl/unthi_zoo/unthi_cifar10_preprocessed/"))
EXPERIMENT_NAME = "sane_cifar10_smallcnnzoo"
WANDB_PROJECT = "sane-cifar10-smallcnnzoo"

def main():
    ### set experiment resources ####
    print(f"torch.cuda.is_available: {torch.cuda.is_available()}")
    # ray init to limit memory and storage
    cpus_per_trial = 18
    gpus_per_trial = 1
    gpus = 1
    cpus = gpus * cpus_per_trial

    # round down to maximize GPU usage

    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpus_per_trial}
    print(f"resources_per_trial: {resources_per_trial}")

    ### configure experiment #########
    experiment_name = EXPERIMENT_NAME

    # set module parameterscd 
    config = {}
    config["seed"] = 32
    config["device"] = "cuda"
    config["device_no"] = 1
    config["training::precision"] = "amp"
    config["trainset::batchsize"] = 512

    config["ae:transformer_type"] = "gpt2" # either "pytorch" or "gpt2"
    config["model::compile"] = True

    # permutation specs
    config["training::permutation_number"] = 5 # any nonzero value of this behaves identically, it is only checked to be == 0 or not
    config["training::view_1_canon"] = True
    config["training::view_2_canon"] = False
    config["testing::permutation_number"] = 5 # any nonzero value of this behaves identically, it is only checked to be == 0 or not
    config["testing::view_1_canon"] = True
    config["testing::view_2_canon"] = False

    config["training::reduction"] = "mean"

    config["ae:i_dim"] = 145
    config["ae:lat_dim"] = 128
    config["ae:max_positions"] = [100, 10, 40] # must be bigger than [window_size, layers, max_channels/filters]
    config["training::windowsize"] = 58
    config["ae:d_model"] = 1024
    config["ae:nhead"] = 8
    config["ae:num_layers"] = 8

    # configure optimizer
    config["optim::optimizer"] = "adamw"
    config["optim::lr"] = 1e-4 # remember to scale with batch size
    config["optim::wd"] = 3e-9
    config["optim::scheduler"] = "OneCycleLR"
    # clip gradients
    config["training::gradient_clipping"] = "norm"
    config["training::gradient_clipp_value"] = 2.0

    # training config
    config["training::temperature"] = 0.1
    config["training::gamma"] = 0.05
    config["training::reduction"] = "mean"
    config["training::contrast"] = "simclr" # can only be "simclr" or "positive", for anything else it will do reconstruction only
    # AMP
    #
    config["training::epochs_train"] = 50
    config["training::output_epoch"] = 5
    # training::test_epochs also influences how frequently results are logged! AE_trainable.step() runs this 
    # amount of training epochs, and then one val/test epoch per step, and ray only updates the results after 
    # each .step() call. If this value is larger than 1, the reported _train metrics are from the _last_ epoch 
    # in the inner loop
    config["training::test_epochs"] = 1
    # development phase: monitor on the val split only, hold out the test split for final evals
    config["training::eval_testset"] = False

    config["monitor_memory"] = True # log memory stats

    # configure output path
    output_dir = OUTPUT_PATH
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        pass

    ###### Datasets ###########################################################################
    DATA_PATH.mkdir(exist_ok=True)
    # path to ffcv dataset for training
    config["dataset::dump"] = DATA_PATH.joinpath("dataset.pt").absolute()
    config["downstreamtask::dataset"] = None
    # call dataset prepper function
    logging.info("prepare data")
    # prep_data(target_dataset_path=DATA_PATH)

    ### Augmentations
    config["trainloader::workers"] = 8
    config["trainset::add_noise_view_1"] = 0.1
    config["trainset::add_noise_view_2"] = 0.1
    config["trainset::noise_multiplicative"] = True # dead key
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
        trial_dirname_creator=lambda trial: f"{RUN_TAG}_{trial.trial_id}_"
        + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    # run
    ray.tune.run_experiments(
        experiments=experiment,
        resume=False,  # resumes from previous run. if run should be done all over, set resume=False
        # resume=True,  # resumes from previous run. if run should be done all over, set resume=False
        reuse_actors=False,
        verbose=3,
        callbacks=[WandbLoggerCallback(project=WANDB_PROJECT)],
    )

    ray.shutdown()
    assert ray.is_initialized() == False


if __name__ == "__main__":
    main()

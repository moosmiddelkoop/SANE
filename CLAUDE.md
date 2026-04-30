# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SANE (Sequential Autoencoder for Neural Embeddings) — research code for the ICML 2024 paper "Towards Scalable and Versatile Weight Space Learning". The package learns task-agnostic representations of neural network *weights* by tokenizing model weights and training a transformer autoencoder over those token sequences. Downstream uses: predicting model properties (test_acc, ggap, epoch) and generating/finetuning new models from sampled embeddings.

## Setup & Commands

Install (uses `uv`, per global convention):
```bash
bash install.sh   # uv pip install -e .  +  ray==2.6.1, pyarrow, imageio
```

Run any script with `uv run` rather than `python` (per global convention). For example:
```bash
uv run experiments/resnet18-cifar100/pretrain_sane_cifar100_resnet18.py
```

Tests use pytest (configured in `setup.cfg` with `--cov SANE`); however `tests/` is not present in the repo, so there is currently no test suite to run.

There is no lint command configured; `setup.cfg` declares `flake8` settings (line length 88, black-compatible) but no pre-commit hook.

## End-to-end pipeline

The pipeline is **always**: download a model zoo → preprocess into tokenized tensors → pretrain SANE autoencoder → run a downstream task (property prediction or sampling/finetuning). Every experiment script assumes the previous stages have been completed and on-disk artifacts exist at hardcoded relative paths. Read the script before running it — paths to pretrained checkpoints (`model_path = Path("path/to/your/model")`) must be filled in by hand.

### 1. Data: model zoos → tokenized datasets
- Zoo download scripts live in `data/` (`download_*.sh`). The CIFAR-10 CNN sample is the smallest and is what the quick-start notebook uses.
- Preprocessing scripts (`data/preprocess_dataset_*.py`) convert a zoo of checkpoints into a `dataset.pt` containing `{"trainset", "valset", "testset"}` of `PreprocessedSamplingDataset`. Key concepts the preprocessor needs:
  - **Permutation spec** (`SANE.git_re_basin.git_re_basin`): describes which weights can be permuted together for the architecture. Use `zoo_cnn_permutation_spec` / `zoo_cnn_large_permutation_spec` / `resnet18_permutation_spec`.
  - **Tokensize / windowsize**: weights are sliced into fixed-size tokens; `windowsize` is the number of tokens per sample. `tokensize=0` means "infer".
  - **Standardize / map_to_canonical**: standardize tokens, and remap permutation-equivalent models to a canonical form for the contrastive objective.
- Vision datasets used to evaluate sampled models live under `data/vision_datasets/` and are prepared by `experiments/*/prepare_*_dataset.py`.

### 2. Pretraining: `AE_trainable` + `AEModule`
- Entry point: `experiments/<zoo>/pretrain_sane_*.py`. These configure a flat `config: dict` (string keys with `::` separators, e.g. `"training::epochs_train"`, `"ae:lat_dim"`) and run it through Ray Tune (`ray.tune.run_experiments`) using `AE_trainable` from `SANE.models.def_AE_trainable`.
- Even when running a single config, the experiment goes through Ray (and writes Ray-style trial dirs / checkpoints under `sane_pretraining/`). To sweep, replace a value with `tune.grid_search([...])`.
- The model is a transformer autoencoder (`SANE.models.def_AE` / `def_transformer.py`, `ae:transformer_type = "gpt2"` by default). Loss is contrastive (`training::contrast = "simclr"`) over two augmented views of weight-token sequences (noise + permutation augmentations from `SANE.datasets.augmentations`).
- Output checkpoints are standard Ray checkpoints. Downstream scripts load them via `AEModule(config)` then `module.model.load_state_dict(checkpoint["model"])` — see `property_prediction_*.py` and `sample_finetune_*.py`.

### 3. Downstream tasks
- **Property prediction** (`experiments/*/property_prediction_*.py`): encodes a `DatasetTokens` of model checkpoints into SANE embeddings and trains baselines from `SANE.models.downstream_baselines` (`IdentityModel`, `LayerQuintiles`) via `DownstreamTaskLearner` (`SANE.models.def_downstream_module`). Writes results to a JSON file.
- **Sampling / finetuning** (`experiments/*/sample_finetune_*.py` and `SANE.sampling.*`): fits a KDE over SANE embeddings (`kde_sample.py` and `_subsampled` / `_bootstrapped` variants), decodes sampled embeddings back to weights, and evaluates the resulting models on the vision dataset. Includes finetune baselines (`finetune_baseline.py`).
- **Ray callbacks during pretraining** (`SANE.evaluation.ray_fine_tuning_callback*`) can periodically sample-and-evaluate models inside a Tune run; they're imported in pretraining scripts but only fire when added to `config["callbacks"]`.

### 4. Quick-start notebook
`experiments/cnn-cifar10_exploration.ipynb` exercises the smallest sample zoo end-to-end (load dataset → instantiate AEModule → encode/decode a model). Use it as the reference for how the pieces fit together.

## Source layout (`src/SANE/`)

- `models/` — autoencoder modules (`def_AE.py`, `def_AE_module.py`), the Ray Tune trainable (`def_AE_trainable.py`), the transformer (`def_transformer.py`), losses (`def_loss.py`), and the downstream learner + baselines.
- `datasets/` — preprocessing (`dataset_preprocessing.py`), the on-disk preprocessed sampler used in training (`dataset_sampling_preprocessed.py`), token/epoch/property dataset variants, and weight-space augmentations.
- `git_re_basin/` — permutation specs and weight-matching utilities (vendored from the Git Re-Basin paper) used to canonicalize weights and as a training augmentation.
- `sampling/` — KDE sampling, decoding embeddings back to model weights, and end-to-end evaluation/finetune routines for sampled models.
- `evaluation/` — Ray Tune callbacks for sample-and-evaluate during training (regular, subsampled, bootstrapped variants).

## Conventions specific to this repo

- **Hardcoded paths.** Experiment scripts contain literal paths like `Path("../../data/dataset_cifar100_token_288_ep60_std/")` and `Path("path/to/your/model")`. They expect to be run from the directory they live in. Don't try to make them portable unless asked — fix the path for the current run instead.
- **Config dict with `::` keys.** All hyperparameters flow through one flat `dict` with namespaced string keys (`"training::..."`, `"ae:..."`, `"optim::..."`, `"trainset::..."`). New options are added by setting a new key on this dict, not by introducing a config class.
- **Ray is mandatory** for pretraining and downstream sampling, even for single-trial runs. CPU/thread env vars are set at the top of every entry script.
- **`.gitignore` excludes `*.pt`, `*.json`, and `/experiments`.** Generated checkpoints, dataset dumps, and result JSONs won't show up in `git status`. Don't be surprised that scripts read/write files that aren't tracked.
- **`install.sh` is currently modified** (per recent git status) — check before running it that it still does the right thing.

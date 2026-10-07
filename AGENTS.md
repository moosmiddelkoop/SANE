# AGENTS.md

## General coding instructions
BE MINIMALIST. I like the code to be as lightweight sa possible. This is code I have to look at and understand fully aswell. Every extra line means more cognitive load. Don't build too many unnecessary abstractions or robustness features. Don't over-engineer. We start simple, and only add complexity when it proves needed. Give the variables names that are good to understand, It's not a big deal if they are long.

## How to communicate with me
Only report to me in ASD-STE100 Simplified Technical English.

- Write every user-facing reply in ASD-STE100 Simplified Technical English.
- Use one idea in each sentence.
- Use a maximum of 20 words in an instruction sentence.
- Use a maximum of 25 words in a descriptive sentence.
- Use a maximum of six sentences in a procedural paragraph.
- Use the active voice.
- Use the simple present tense when possible.
- Keep the articles "the" and "a".
- Use one word for one meaning.
- Do not replace a word with a synonym for variety.
- Do not use idioms, slang, or figures of speech.
- Keep technical names unchanged. This includes files, commands, functions, classes, variables, and error text.
- Use plain language.
- Explain an unavoidable technical term with a short definition.
- Lead with the action or the outcome.
- Start a completed task with:
  "Done: <outcome>"
- Do not add a conversational preamble.
- Do not start with phrases such as:
  "Let me..."
  "Great question..."
  "I would be happy to..."
  "Based on your request..."
- Put the context and reasoning after the action.
- Use numbered steps for a sequence.
- Put one bounded action in each step.
- Use a maximum of five items in one list.
- Split a longer list into:
  "Do now"
  and
  "Do later"
- Restate the task state during every turn of a multi-step task.
- Use this format:
  "Step 3 of 5 done: schema updated. Next: backfill."
- Do not assume that the user remembers the previous message.
- Give a concrete time estimate when the task requires user work.
- Do not use vague estimates such as:
  "This will take some work."
- Keep normal answers to six sentences or fewer unless the user asks for depth.
- Answer only the requested topic.
- Do not include unrequested alternatives, comparisons, or tangents.
- End with one concrete next action when work remains.
- Do not end with:
  "Let me know."
  "Tell me what you think."
  "I can help with that."
- State assumptions before you act.
- Ask a question only when a requirement is genuinely ambiguous.
- Otherwise, select the sensible default and state the selected default.
- If a second issue appears, finish the first issue.
- Offer the second issue as a separate task.
- Do not combine the second issue with the current task.
- After a change, summarize:
  - What changed
  - Where it changed
  - Why it changed
- Include exact file paths when files change.
- After a feature change, add a short manual test checklist.
- The checklist must state what to open, click, enter, and confirm.

## Project

SANE (Sequential Autoencoder for Neural Embeddings) — research code for the ICML 2024 paper "Towards Scalable and Versatile Weight Space Learning". The package learns task-agnostic representations of neural network *weights* by tokenizing model weights and training a transformer autoencoder over those token sequences. Downstream uses: predicting model properties (test_acc, ggap, epoch) and generating/finetuning new models from sampled embeddings.

The accompanying paper can be found in `paper/` (both .tex source, .md and .pdf versions). I am trying to use SANE for my weight space unlearning work (see paper in `CNNZoo_surgery_ICML2026-13.pdf`), but that code is not yet in this repo. The SANE side of that work (preprocessing, pretraining and per-class recall prediction on the small CNN "unthi" zoos) is in this repo — see "Small CNN zoo (unthi) workflow" below.

Tests use pytest (configured in `setup.cfg` with `--cov SANE`); however `tests/` is not present in the repo, so there is currently no test suite to run.

There is no lint command configured; `setup.cfg` declares `flake8` settings (line length 88, black-compatible) but no pre-commit hook.

## End-to-end pipeline

The pipeline is **always**: download a model zoo → create its fixed split (`data/create_split.py`) → preprocess into tokenized tensors → pretrain SANE autoencoder → run a downstream task (property prediction or sampling/finetuning). Every experiment script assumes the previous stages have been completed and on-disk artifacts exist at hardcoded relative paths. Read the script before running it — paths to pretrained checkpoints (`model_path = Path("path/to/your/model")`) must be filled in by hand.

### 1. Data: model zoos → tokenized datasets
- Zoo download scripts live in `data/` (`download_*.sh`). The CIFAR-10 CNN sample is the smallest and is what the quick-start notebook uses.
- Preprocessing scripts (`data/preprocess_dataset_*.py`) convert a zoo of checkpoints into a consolidated `dataset.pt` containing `{"trainset", "valset", "testset"}` of `TensorSamplingDataset` (all samples stacked into in-RAM tensors; one sequential read at training time) via `SANE.datasets.dataset_preprocessing_consolidated`. The original variant (`dataset_preprocessing.py`: per-sample `.pt` files in `dataset_torch.{split}/` dirs, wrapped by path-based `PreprocessedSamplingDataset`) still exists — switch the entry script's import to use it. Legacy per-sample zoos can be converted with `data/consolidate_preprocessed.py`. Key concepts the preprocessor needs:
  - **Permutation spec** (`SANE.git_re_basin.git_re_basin`): describes which weights can be permuted together for the architecture. Use `zoo_cnn_permutation_spec` / `zoo_cnn_large_permutation_spec` / `resnet18_permutation_spec`.
  - **Tokensize / windowsize**: weights are sliced into fixed-size tokens; `windowsize` is the number of tokens per sample. `tokensize=0` means "infer".
  - **Standardize / map_to_canonical**: standardize tokens, and remap permutation-equivalent models to a canonical form for the contrastive objective.
- Unlike the other `preprocess_dataset_*.py` scripts, `data/preprocess_dataset_smallcnnzoo.py` takes `--in_dir` / `--out_dir` so one script serves all four small CNN zoos; `data/preprocess_all_unthi.sh` runs it for all of them (see "Small CNN zoo (unthi) workflow").
- Vision datasets used to evaluate sampled models live under `data/vision_datasets/` and are prepared by `experiments/*/prepare_*_dataset.py`.

### 2. Pretraining: `AE_trainable` + `AEModule`
- Entry point: `experiments/<zoo>/pretrain_sane_*.py`. These configure a flat `config: dict` (string keys with `::` separators, e.g. `"training::epochs_train"`, `"ae:lat_dim"`) and run it through Ray Tune (`ray.tune.run_experiments`) using `AE_trainable` from `SANE.models.def_AE_trainable`.
- Even when running a single config, the experiment goes through Ray (and writes Ray-style trial dirs / checkpoints under `sane_pretraining/`). To sweep, replace a value with `tune.grid_search([...])`.
- The model is a transformer autoencoder (`SANE.models.def_AE` / `def_transformer.py`, `ae:transformer_type = "gpt2"` by default). Loss is contrastive (`training::contrast = "simclr"`) over two augmented views of weight-token sequences (noise + permutation augmentations from `SANE.datasets.augmentations`).
- Output checkpoints are standard Ray checkpoints. Downstream scripts load them via `AEModule(config)` then `module.model.load_state_dict(checkpoint["model"])` — see `property_prediction_*.py` and `sample_finetune_*.py`.

### 3. Downstream tasks
- **Property prediction** (`experiments/*/property_prediction_*.py`): encodes a `DatasetTokens` of model checkpoints into SANE embeddings and trains baselines from `SANE.models.downstream_baselines` (`IdentityModel`, `LayerQuintiles`) via `DownstreamTaskLearner` (`SANE.models.def_downstream_module`). Writes results to a JSON file.
- **Per-class recall heads** (`DownstreamTaskLearner`, used by the small CNN zoo scripts): `eval_per_class_recall_multivariate_regression` (renamed from `eval_multivariate_regression`) fits one closed-form ridge head with K outputs. `eval_per_class_recall_MLP` trains an MLP head (`D → 128 → 128 → K`, ReLU, Adam, MSE loss) and returns `final_train_loss`, `mse_*`, `mae_*`, mean `r2_*` for train and test, plus the trained `mlp`. It accepts either a dataset (it encodes it) or a precomputed `(embeddings, targets)` tuple (from an embeddings cache; pass `model=None`). It maps the `-999` sentinel to NaN and drops rows with any NaN target. An optional `log_fn` (e.g. `wandb.log`) receives `loss_step`, `loss_epoch` and `test_loss_epoch`; the caller owns the logging backend.
- **Sampling / finetuning** (`experiments/*/sample_finetune_*.py` and `SANE.sampling.*`): fits a KDE over SANE embeddings (`kde_sample.py` and `_subsampled` / `_bootstrapped` variants), decodes sampled embeddings back to weights, and evaluates the resulting models on the vision dataset. Includes finetune baselines (`finetune_baseline.py`).
- **Ray callbacks during pretraining** (`SANE.evaluation.ray_fine_tuning_callback*`) can periodically sample-and-evaluate models inside a Tune run; they're imported in pretraining scripts but only fire when added to `config["callbacks"]`.

### 4. Quick-start notebook
`experiments/cnn-cifar10_exploration.ipynb` exercises the smallest sample zoo end-to-end (load dataset → instantiate AEModule → encode/decode a model). Use it as the reference for how the pieces fit together.

## Source layout (`src/SANE/`)

- `models/` — autoencoder modules (`def_AE.py`, `def_AE_module.py`), the Ray Tune trainable (`def_AE_trainable.py`), the transformer (`def_transformer.py`), losses (`def_loss.py`), and the downstream learner + baselines.
- `datasets/` — preprocessing (`dataset_preprocessing_consolidated.py`, default; `dataset_preprocessing.py`, per-sample-file variant), the samplers used in training (`dataset_sampling_preprocessed.py`: `TensorSamplingDataset` and `PreprocessedSamplingDataset`), token/epoch/property dataset variants, and weight-space augmentations.
- `git_re_basin/` — permutation specs and weight-matching utilities (vendored from the Git Re-Basin paper) used to canonicalize weights and as a training augmentation.
- `sampling/` — KDE sampling, decoding embeddings back to model weights, and end-to-end evaluation/finetune routines for sampled models.
- `evaluation/` — Ray Tune callbacks for sample-and-evaluate during training (regular, subsampled, bootstrapped variants).
- `utils.py` — `seed_everything(seed)`: seeds `random`, numpy, torch (CPU + CUDA), sets `PYTHONHASHSEED`, and forces deterministic cuDNN. The only seeding helper in the package (it moved here from `def_AE_module.py`).

## Small CNN zoo (unthi) workflow

The zoo family used for the unlearning work. Everything runs on the Snellius HPC cluster via SLURM.

- **Zoos.** Four image datasets — MNIST, Fashion-MNIST (`fmnist`), CIFAR-10, SVHN — with 30,000 models each and 9 checkpoints per model (epochs 0–8). All share one architecture (3 conv layers + dense head), so they share one permutation spec (`smallcnnzoo_permutation_spec`) and one token geometry (`tokensize=145`, `windowsize=58`). The raw directory layout is in `CWI-README.md`.
- **Storage.**
  - Raw zips: `/projects/prjs2156/shared/wsl/unthi_zoo/unthi_<zoo>.zip`. `/projects` is near its inode quota, so extract to `/gpfs/scratch1/shared/mmiddelkoop/unthi_zoo/` (same as `/scratch-shared/mmiddelkoop/unthi_zoo/`). Scratch deletes files that are untouched for two weeks.
  - Preprocessed: `/projects/prjs2156/shared/wsl/unthi_zoo/unthi_<zoo>_preprocessed/dataset.pt`. MNIST exception: the stacked dataset is `…/unthi_mnist_preprocessed/consolidated/dataset.pt`.
  - Pretrained encoders: `/projects/prjs2156/shared/wsl/metanets/sane_pretraining/<zoo>/<RUN_TAG>_<trial_id>_<date>/checkpoint_0000NN/state.pt`. Old MNIST runs are in `legacy/`. The encoders used downstream are `mnist-v2.0_6133f…`, `cifar10-v1.0_d496b…`, `fmnist-v1.0_d3e0b…`, `svhn-v1.0_d4950…`, all at `checkpoint_000050`.
- **Preprocess.** `data/preprocess_all_unthi.sh` (sbatch, CPU partition `rome`, about 5 h for all four zoos) unzips each zoo to scratch, runs `preprocess_dataset_smallcnnzoo.py --in_dir=… --out_dir=…`, and deletes the extracted copy. Settings: epochs 0–8, `ds_split=[0.7, 0.15, 0.15]`, `shuffle_path=True`, `map_to_canonical=True`, `standardize=True`, `ignore_bn=True`, `weight_threshold=100`. Memory note (stacking workers capped at 8, ~50 GB peak for a 30k-model zoo) is in `README.md`.
- **Pretrain.** `experiments/smallcnnzoo-<zoo>/pretrain_sane_<zoo>_smallcnnzoo.py`, submitted with the sibling `.sh` (`gpu_a100`, 1 GPU, 18 CPUs). The `.sh` sets `SANE_DATA_DIR`, which overrides the script's `DATA_PATH`. Output naming: `OUTPUT_PATH / EXPERIMENT_NAME (= zoo name) / f"{RUN_TAG}_{trial_id}_…"`; `RUN_TAG` also names the W&B run in project `sane-pretraining-smallcnnzoo`. Bump the `RUN_TAG` version (`mnist-v2.0`, `cifar10-v1.0`, …) when the launch config changes. Gradient clipping is on (`training::gradient_clipping = "norm"`, `training::gradient_clipp_value = 2.0`) since the loss spike in run `c898d` (`reports/loss-spike-postmortem-c898d.md`, `TODO.md`). `AEModule` logs `debug/z_norm`, `debug/z_var`, `debug/skipped_steps`, `debug/clipped_steps`, `debug/pre_clip_norm_{max,median}` per epoch. MNIST trains 200 epochs, the others 50.
- **Target property: per-class recall** `acc_class_0..9`. It is only valid at training iterations 0, 4 and 8; other iterations hold the sentinel `-999`.
- **Linear-head analyses (MNIST only)** in `experiments/smallcnnzoo-mnist/`: `property_prediction_mnist_smallcnnzoo.py` (univariate ridge) and `_multivariate.py` (one K-output ridge head); `recall_prediction_r2_spread.py` / `recall_prediction_mse_spread.py` (spread over 10 re-splits for epoch sets `[0,4,8]`, `[4,8]`, `[8]`); `recall_prediction_canon_jitter.py` (spread from canonicalization alone); `compare_spike_checkpoints.py` (pre- vs post-spike encoder); `plot_*.py` (figures in `recall_prediction/`). Each has a sibling `.sh`.
- **MLP-head runs (all four zoos)**: `experiments/smallcnnzoo-<zoo>/property_prediction_<zoo>_smallcnnzoo_mlp.py` + `.sh` (`rome`, 16 CPUs, 8 h). The script:
  1. Encodes epoch 8 of the raw zoo (`ZOO_ROOT` on scratch) with the encoder's `checkpoint_000050`. It uses `shuffle_path=False`, so model dirs are in sorted order: the first 80 % are train, the last 20 % test (`DS_SPLIT = [0.8, 0.2]`).
  2. Caches the result in `recall_prediction/mlp/embeddings_sorted.pt` as `{EPOCH_SET: {"z", "Y", "mid", "model_dirs"}}` (`mid` = model id per sample). With `USE_EMBEDDINGS_CACHE = True` and an existing cache, step 1 is skipped.
  3. For each seed in `SEEDS`, trains `eval_per_class_recall_MLP` on the cached split. `RESHUFFLE` (MNIST) / `RE_SHUFFLE` (other zoos) reshuffles the model ids per seed before the split; it is `False` in the final runs.
  4. Logs each seed to W&B project `sane-per-class-recall-mlp`, saves `mlp_head_seed{seed}.pt`, and writes a results JSON with `per_seed`, `mean` and `std` (std is `0.0` for a single seed).
  - Final settings: MNIST uses 10 seeds (0–9) and 100 epochs; CIFAR-10, Fashion-MNIST and SVHN use seed `[67]` and 200 epochs.
- **Split recovery.** Older MLP caches (`embeddings.pt`, random 3-way split, `random.Random(67)`) have no `model_dirs`. `experiments/recover_recall_mlp_split.py` (+ `.sh`) recovers which model dirs were in train/val/test: each target row `Y` matches line 8 of one model's `result.json`, and the script replays the shuffle. It writes `recall_prediction/mlp/split.json` per zoo (a list entry = ambiguous match, `null` = no match). The current caches store `model_dirs`, so this is not needed for new runs.
- **Reports.** `reports/loss-spike-postmortem-c898d.md` (spike cause: no clipping near the OneCycleLR peak) and `reports/contrastive-val-loss-discrepancy.md` (val/test NT-Xent depends on batch composition, so a data-order change moves it without a change in model quality).

## Conventions specific to this repo

- **Hardcoded paths.** Experiment scripts contain literal paths like `Path("../../data/dataset_cifar100_token_288_ep60_std/")` and `Path("path/to/your/model")`. They expect to be run from the directory they live in. Don't try to make them portable unless asked — fix the path for the current run instead.
- **Config dict with `::` keys.** All hyperparameters flow through one flat `dict` with namespaced string keys (`"training::..."`, `"ae:..."`, `"optim::..."`, `"trainset::..."`). New options are added by setting a new key on this dict, not by introducing a config class.
- **Ray is mandatory** for pretraining and downstream sampling, even for single-trial runs. CPU/thread env vars are set at the top of every entry script.
- **`.gitignore` excludes `*.pt`, `*.json`, and `/experiments`.** Generated checkpoints, dataset dumps, and result JSONs won't show up in `git status`. Don't be surprised that scripts read/write files that aren't tracked.
- **`install.sh` is currently modified** (per recent git status) — check before running it that it still does the right thing.
- **Every zoo has one fixed split: `<zoo>/split.json`.** Create it once, before anything else, with `uv run data/create_split.py <zoo>` (`SANE.datasets.zoo_split`). All dataset classes take their train/val/test models from this file; there is no `ds_split` or `shuffle_path` anymore, and preprocessing rejects configs that still set them. Without the file, every dataset class fails with `MissingSplitError`. The file is never overwritten: `create_split` refuses if it exists, and `load_split` fails if it was edited or if the zoo's model directories changed. Each split has a `split_id`, stored in `dataset_info_<split>.json` and on the datasets inside `dataset.pt`. Pretraining (`AE_trainer.load_datasets`) checks that train/val/test share one split and do not overlap; downstream scripts call `assert_same_split(config, dataset)` to check that the zoo's split is the one the encoder was pretrained on. Data made before this existed has no `split_id` and fails both checks; set `config["dataset::legacy_unverified_split"] = True` to use it anyway, knowingly. `train_val_test="all"` loads every model without a split file, for sampled populations only. Cross-validation and split-variance experiments may re-split train and val with `resplit_train_val(train, val, seed)`; the test split is never re-split.
- **Downstream encoding must match pretraining preprocessing.** Property-prediction and sampling scripts also load checkpoints through `DatasetTokens` and re-apply `map_to_canonical`, the permutation spec, `ignore_bn`, and the standardization stats before encoding. These must match the values used at preprocessing time — otherwise the embeddings will be off-distribution and downstream metrics will silently degrade. The epoch list itself is *not* part of this constraint (it just picks which checkpoints to encode), but every other preprocessing param is.
- **Epoch list convention (cnn-cifar10): stride-5 sweep `[5, 10, …, 50]`.** This repo's other preprocess scripts use either a single late epoch (`[60]`) or a tight late-window (`[21..25]`). The CNN-CIFAR10 preprocessing in `data/preprocess_dataset_cnn_cifar10.py` deliberately deviates from both — it samples 10 checkpoints across the full training trajectory at stride 5. The motivation is the weight-space *unlearning* downstream use case: unlearning perturbs a converged model and can push it off the converged manifold towards earlier-training-like states, and we want SANE embeddings to remain trustworthy over that broader region (a "high trust region" spanning the training trajectory, not just the converged endpoint). For purely converged-state work this is overkill; for unlearning / model-editing work it is the right shape.
- **Seeding: one knob, set once at the entry point.** Every entry script must (1) import `from SANE.utils import seed_everything`, (2) call `seed_everything(SEED)` before any dataset construction or model init, and (3) set `config["seed"] = SEED` so `AEModule`/`NNmodule` re-seed Ray workers from the same value. Library code must never call `random.seed(...)`/`np.random.seed(...)`/`torch.manual_seed(...)` with a hardcoded literal — it would override the entry-point seed mid-run. The only seeding call inside library code is `seed_everything(config.get("seed", 42))` in `AEModule.__init__` and `NNmodule.__init__`, which is idempotent when the entry script already called it and is what seeds fresh Ray worker processes.
- **Splits follow the entry-point seed.** The hardcoded `random.seed(42)` calls were removed from `dataset_epochs.py`, `dataset_properties.py`, and `sample_finetune_auxiliaries.py` (2026-07-30). With `shuffle_path=True`, the path shuffle — and so the train/val/test split — now uses the global `random` state that the entry script seeds. Datasets and results made before 2026-07-30 used seed 42 (older docstrings still say "seed-42 split"). The `data/preprocess_dataset_*.py` scripts do not call `seed_everything`.
- **`shuffle_path=False` gives a deterministic, sorted split.** `ModelDatasetBaseEpochs` then sorts the model dirs by `(len(name), name)` — numeric order for unpadded numeric names — so the first `ds_split[0]` fraction is always train, with no RNG involved.
- **Canonicalization is stochastic.** git-re-basin `weight_matching` visits permutation blocks in `torch.randperm` order inside Ray workers that `seed_everything` does not reach. The canonical form, and so the embeddings, vary slightly between dataset builds even with an identical split. `recall_prediction_canon_jitter.py` measures this jitter.
- **`DatasetTokens(epoch_lst=...)` takes a list** (default `[10]`).
- **Consolidation (the in-RAM `dataset.pt`).** Why: the per-sample flow re-read 212k files (43 GB, unthi MNIST) each epoch, about 12 epochs in 12 h. History: 2026-07-03 `TensorSamplingDataset` + `data/consolidate_preprocessed.py` (converts an existing per-sample zoo into `<zoo>/consolidated/dataset.pt`; MNIST went this way); 2026-07-14 `dataset_preprocessing_consolidated.py` writes the stacked `dataset.pt` directly (CIFAR-10, Fashion-MNIST, SVHN went this way), then the OOM fix that caps stacking workers at 8. Consequences:
  - The location differs per zoo (MNIST: `consolidated/` subdir; others: the root). Check it before you set `SANE_DATA_DIR`. The pretrain scripts' `DATA_PATH.mkdir(exist_ok=True)` silently creates an empty directory for a wrong path; on 2026-07-15 this made three launches fail.
  - A pickled per-sample `dataset.pt` contains absolute sample paths, so a staging copy of the per-sample files is never read.
  - Consolidated stores samples model-major (the 9 checkpoints of a model are consecutive), and val/test loaders use `shuffle=False`. So each val/test batch holds near-duplicate same-model negatives and NT-Xent val/test loss is much higher. It is **not comparable across the consolidation boundary** (e.g. `e550b` vs `c898d`); compare downstream R² instead. Train loss is not affected (`shuffle=True`). The proposed fix — one fixed seeded permutation of val/test via `torch.utils.data.Subset` in `def_AE_trainable.load_datasets` — is **not applied**. Details: `reports/contrastive-val-loss-discrepancy.md`.
  - If a dataset outgrows RAM, the plan is `torch.load(mmap=True)`, then sharded streaming.

# Code for "Utilizing Weight Space Learning for Data-Free Model Editing" (TMLR 2026 submission)

This directory holds the code behind the experiments of the paper. Every file is a full copy of
the file that was run. Only path, scheduler and contact lines were replaced by placeholders
(see "Changes to copied files").

| Folder | Zoo | Source |
|---|---|---|
| `small_mlp_blob_zoo/` | Small MLP: 2D Gaussian blob zoo | code export (manifest) |
| `small_cnn_zoo/` | Small CNN zoo (MNIST, Fashion-MNIST, CIFAR-10, SVHN) | code export (manifest) |
| `resnet18_cifar100_recall_labels/` | ResNet-18 CIFAR-100 zoo: download, unpack, per-class recall labels, Table 3 | SANE-based codebase |
| `resnet18_cifar100_zoo/` | ResNet-18 CIFAR-100 zoo: metanetworks, edits, baselines | code export (manifest) |

`code_export_manifest.csv` records where each file of the code export comes from, with
its SHA-256 before and after path anonymization (`path_substitutions` = number of lines changed,
`byte_identical` = no change). All 98 files match their `sha256_in_zip`.

Plotting code, model-zoo weights, trained metanetwork checkpoints and result files are not included.
Each folder is flat because the scripts import their siblings by module name; run them with the
folder as working directory.

## Paper item → code

| Paper item | Blob zoo | Small CNN zoo | ResNet-18 zoo |
|---|---|---|---|
| Zoo and recall labels (App. A, Tables 1–3) | `toyzoo.py`, `gen_zoo.py` | `cnnzoo.py`, `build_zoo.py`, `prep_zoo.py`, `stage_zoo.py` | `resnet18_cifar100_recall_labels/`, then `eval_epochs.py`, `resnet_zoo.py`, `stage_zoo_epoch.py` |
| Edit sets (Table 6) | `ublib.py` (`edit_set`) | `build_zoo.py`, `prep_zoo.py` | `make_edit_sets.py` |
| MLP metanetwork (Sec. 5.2.1, App. B.1) | `exp.py`, `worker.py`, `pool.py` | `meta_cnn.py`, `run_meta.py` | not applicable |
| GMN (Sec. 5.2.2, App. B.2, Table 4) | `exp.py`, `worker.py` | `meta_cnn.py`, `run_meta.py` | `resnet_gmn.py`, `train_gmn.py`, `gmn_psi.py` |
| SANE + head (Sec. 5.2.3, App. B.3, Table 5) | `sane_toy_pretrain_blobs.py`, `sane_psi.py`, `sane_vendor/` | `sane_zoo.py`, `run_sane_pretrain.py`, `sane_vendor/` | `sane_resnet.py`, `build_tokens_ep20.py`, `run_pretrain.py`, `eval_recon_test.py`, `zoo_features.py`, `fit_heads*.py`, `sane_vendor/` |
| Ridge baseline, interaction R²× (Figs. 3, 21; Table 11) | `blobs_preds.py`, `blobs_score.py` | `baselines_meta.py`, `class_centred_r2.py` | `fit_heads_ep20.py` |
| Algorithm 1, ψ-stop, oracle stop (Figs. 2, 20; Tables 7–9) | `ublib.py`, `run_walk.py` | `edit_cnn.py`, `run_edit.py`, `free_walk.py` | `edit_resnet.py`, `run_edit_gmn.py`, `run_edit_sane_gmn.py`, `jobs/` |
| GA, FT baselines (Table 10) | `run_data.py` | `run_edit.py` | `run_edit_gmn.py` |
| RV, Bias controls | `post_edit.py`, `add_controls.py`, `regen_controls.py` | `make_controls.py`, `make_controls_fw.py`, `oracle_path.py` | `build_layer_rv_ep20.py` |
| Faithfulness (Fig. 4) | `run_walk.py` | `full_walk.py` | `run_edit_gmn.py --fixed_steps` |
| Relearning attack (Fig. 5) | `ublib.py` (`relearn`), `post_edit.py`, `add_controls.py` | `relearn_cnn.py` | `relearn_resnet_fixidx.py` |
| Where the edits land (Fig. 6) | `ublib.py` (`block_shares`), `post_edit.py` | `edit_cnn.py`, `run_edit.py`, `psi_rewalk.py` | `run_edit_gmn.py`, `run_edit_sane_gmn.py`, `analyze_edits.py` |
| Output-layer edits (App. F, Fig. 22, Table 12) | `run_walk.py <spec> output` | `run_edit.py meta_out:<run>` | `run_edit_gmn.py meta_fc:gmn`, `run_edit_sane_gmn.py meta_fc:sane_token_last` |
| Gradient directions (Fig. 7) | `dir_walk.py`, `ga_norm.py`, `dir_analysis.py` | – | – |

## Settings shared by all zoos (as used for the figures)

Algorithm 1 with normalised step η = 1e-3 · ‖θ₀‖, Boost loss with β = 0.1, relative ψ-stop ψ_c ≤ 0.4 · ψ_c(θ₀) (60 % predicted drop), step cap 2,000 (blob and Small CNN zoos) or 1,000 (ResNet-18). The oracle stop uses the same 0.4 factor on true target-class recall. These values are the defaults / constants in `ublib.py` (`ETA, BETA, REL, MAXS`), `edit_cnn.py` and the command lines in `resnet18_cifar100_zoo/jobs/`.

## `small_mlp_blob_zoo/` — Small MLP: 2D Gaussian blob zoo

| Step | Files | Command |
|---|---|---|
| Zoo (50,000 MLPs, 2-16-16-6) | `toyzoo.py`, `gen_zoo.py` | `python gen_zoo.py 50000 600 zoo_50k.npz` (the stored zoo was produced by the same `toyzoo.train_zoo(50000, seed=777, steps=600)` call and the same seed-0 split, without the `task` key) |
| MLP and GMN metanetworks | `exp.py` (`MLPMeta`, `NeuronNet`, training loop), `worker.py`, `pool.py` | `ZOO=zoo_50k.npz python worker.py mlp_raw <seed> 60` and `python worker.py neuron_r1 <seed> 60` |
| SANE + token head | `sane_toy_pretrain_blobs.py`, `sane_vendor/` | `python sane_toy_pretrain_blobs.py --zoo zoo_50k.npz --seed <seed>`, with an 8-minute training budget (`--train_min`) |
| Ridge baseline, prediction collection, scoring | `blobs_preds.py` (ridge on weight statistics; imports `meta.py` for a legacy row), `blobs_score.py` (interaction R² and bootstrap CI) | `python blobs_preds.py` |
| Algorithm 1 edits (ψ-stop, oracle stop, faithfulness path) | `ublib.py` (protocol), `run_walk.py`, `sane_psi.py` (SANE as ψ) | `python run_walk.py <mlp_raw\|neuron_r1\|sane> <full\|output>` |
| GA and FT baselines | `run_data.py` | `python run_data.py` |
| Per-edit results, layer shares, relearning, RV/Bias controls | `post_edit.py`, `add_controls.py`, `regen_controls.py` | `python post_edit.py`, `python add_controls.py` |
| Gradient directions | `dir_walk.py`, `ga_norm.py` (fixed-step GA), `dir_analysis.py` | `python dir_walk.py <spec> <seed> 300`, `python ga_norm.py` |
| Consistency checks | `check_nets.py`, `check_sane.py` | |

## `small_cnn_zoo/` — Small CNN zoo (MNIST in the main text; the same code runs the other three collections)

| Step | Files | Command |
|---|---|---|
| Zoo assembly and edit sets | `cnnzoo.py` (network reconstruction, per-class recall), `build_zoo.py`, `prep_zoo.py`, `stage_zoo.py` | `python build_zoo.py mnist <zoo_dir>`, `python prep_zoo.py mnist` |
| MLP and GMN metanetworks | `meta_cnn.py` (`MLPMeta`, `ChannelGraphNet`, invariant features, training), `run_meta.py` | `python run_meta.py mlp_paper <seed> 30`, `python run_meta.py graph1 <seed> 30` |
| Ridge baseline | `baselines_meta.py` | `python baselines_meta.py` |
| Interaction R² | `class_centred_r2.py` | imported |
| SANE + heads | `sane_zoo.py`, `run_sane_pretrain.py`, `sane_vendor/` | `python run_sane_pretrain.py <seed> 50` |
| Algorithm 1 edits with ψ-stop, GA, FT, RV/Bias | `edit_cnn.py`, `run_edit.py` | `python run_edit.py meta:<run> unlearn --n_per_class 200 --max_steps 2000 --log_every 20 --judges <runs> --save_delta --traj_all`; `python run_edit.py ga_real unlearn --save_delta`; `python run_edit.py ft_retain unlearn --save_delta` |
| Oracle stop (free walk, no ψ-stop) | `free_walk.py`, `make_controls_fw.py` | `python free_walk.py --edits edits.npz --run <run> --key meta --state_dir DIR --classes <list> --cap 10000` |
| Faithfulness walk (no stop) | `full_walk.py` | `python full_walk.py --edits edits.npz --run <run> --key meta --state_dir DIR --classes <list>` |
| Per-metanetwork RV/Bias controls | `make_controls.py`, `oracle_path.py` | see the docstrings |
| Relearning attack | `relearn_cnn.py` | `python relearn_cnn.py results/<tag>.npz` |
| Layer shares of the SANE edit | `psi_rewalk.py` | see the docstring |
| Helpers | `pick_eta_z.py`, `compact_npz.py` | |
| Job templates as submitted | `jobs/*.sh` | full command sequences, with `__PLACEHOLDER__` fields |

`<run>` is `graph1_s0_e30`, `mlp_paper_s0_e30` or `sane_s0_e50@token`; seed-1 runs act as judges.

## ResNet-18 CIFAR-100 zoo (epoch-20 checkpoints)

### `resnet18_cifar100_recall_labels/` — run this first

The scripts of `resnet18_cifar100_zoo/` read the per-class recall `acc_class_0..99` from the
`result.json` of every zoo model. The scripts in this folder download the zoo and write these keys.

| Step | Files | Command |
|---|---|---|
| Download the full zoo (~2.8 TB tar) | `data/download_resnet18_cifar100_full.sh` | `sbatch data/download_resnet18_cifar100_full.sh` |
| Unpack every 5th epoch (0, 5, …, 60) | `data/unpack_resnet18_cifar100_full.sh` | `sbatch data/unpack_resnet18_cifar100_full.sh` |
| CIFAR-100 (torchvision download) | `data/create_cifar100_dataset.py` | `python create_cifar100_dataset.py` in the parent directory of `CIFAR100_ROOT`; it also writes a train/val split to `CIFAR100/dataset.pt`, which the recall labels do not use |
| Per-class recall into `result.json`, check against the logged `test_acc` | `evaluate_per_class_recall_cifar100_resnet18.py`, `SANE/` (`ResNet18` from `def_net.py`, `seed_everything` from `utils.py`) | `sbatch evaluate_per_class_recall_cifar100_resnet18.sh` |
| Table 3 statistics (and a class-rank figure that is not in the paper) | `plot_class_rank_epochs.py` | `python plot_class_rank_epochs.py` |

`SANE/` holds four full files of the `src/SANE/` package of the SANE-based codebase. Python finds them first when a script of
this folder runs, so an installed `SANE` package is not needed.

### `resnet18_cifar100_zoo/` — metanetworks and edits

| Step | Files | Command |
|---|---|---|
| Per-class recall of the zoo, staging, edit sets | `eval_epochs.py`, `resnet_zoo.py`, `stage_zoo_epoch.py`, `make_edit_sets.py` | `jobs/job_stage.sh` |
| GMN | `resnet_gmn.py`, `train_gmn.py`, `gmn_psi.py` | `python train_gmn.py ROOT OUT --seed <seed> --epochs 300` (`jobs/job_train_template.sh`) |
| SANE autoencoder | `sane_resnet.py`, `build_tokens_ep20.py`, `run_pretrain.py`, `eval_recon_test.py`, `sane_vendor/` | `jobs/job_D_sane_pretrain_ep20.sh` |
| SANE heads and ridge baselines | `zoo_features.py`, `fit_heads_ep20.py` (extends `fit_heads.py`) | `jobs/job_E1_sane_heads_ep20.sh` |
| Algorithm 1 edits, GMN | `edit_resnet.py`, `run_edit_gmn.py` | `python run_edit_gmn.py meta:gmn unlearn --n_per_class 6 --beta 0.1 --eta 1e-3 --max_steps 1000 --log_every 50 --batch 40 --judges gmn@s<judge seed> --delta_full_n 1`; oracle stop: `--max_steps 200 --log_every 5 --oracle_stop`; faithfulness walk: `--max_steps 100 --log_every 5 --fixed_steps` (`jobs/job_edit_template.sh`, `jobs/job_oracle.sh`, `jobs/job_oracle2.sh`) |
| GA and FT baselines | `run_edit_gmn.py` | `python run_edit_gmn.py ga_real unlearn --n_per_class 2 --lr_mult 1e-3 --ga_steps 200 --batch 40`; `python run_edit_gmn.py ft_retain unlearn --n_per_class 2 --lr_mult 1e-3 --ft_steps 100 --batch 40` |
| Algorithm 1 edits, SANE | `run_edit_sane_gmn.py` | `python run_edit_sane_gmn.py meta:sane_token_full unlearn --n_per_class 6 --beta 0.1 --eta 1e-3 --max_steps 1000 --log_every 50 --batch 8 --judges gmn@s3 --delta_full_n 1` (`jobs/job_E3a*.sh`, `jobs/job_E3b*.sh`) |
| Relearning attack | `relearn_resnet_fixidx.py` (corrected version of `relearn_resnet.py`) | `jobs/job_R_gmn_relearn_fix_ep20.sh` |
| Per-layer norm-matched RV control | `build_layer_rv_ep20.py` | `jobs/job_LRV_*.sh` |
| Result tables | `analyze_edits.py`, `compact_results.py` | |

`train_mlpfc_ep20.py` and `mlpfc_psi.py` belong to the same code directory (flat MLP on the output layer only); they are not used for any figure.

## Changes to copied files

- Code export: file-system paths, scheduler partition names and one cluster name in a comment were replaced by placeholders (`/path/to/...`, `<gpu_partition>`). No other line was changed. The manifest lists the changed files.
- `resnet18_cifar100_recall_labels/`: the same kind of placeholders (`/path/to/...`, `<gpu_partition>`, `<staging_partition>`, `<email>`). Changed lines per file: `evaluate_per_class_recall_cifar100_resnet18.sh` 5 (partition, log path, e-mail, `cd` into this folder, virtual environment), `evaluate_per_class_recall_cifar100_resnet18.py` 2 (`ZOO_ROOT`, `CIFAR100_ROOT`), `plot_class_rank_epochs.py` 1 (`ZOO_ROOT`), `data/download_resnet18_cifar100_full.sh` 4 and `data/unpack_resnet18_cifar100_full.sh` 4 (partition, log path, e-mail, zoo path). The other five files are unchanged. Fill in the placeholders before a run.

## Notes

- Job scripts in `jobs/` are the submitted Slurm scripts or their templates. They unpack a code tarball and therefore call the scripts with a `code/` or `gmn_code/` prefix; in this directory the same files sit directly in the zoo folder.
- `sane_vendor/` holds three files of the public SANE reference implementation (MIT licence); see `sane_vendor/NOTICE.md`.
- Software used for the CPU runs of the code export: Python 3.12, PyTorch 2.13.0, NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1 (pandas 3.0.6 and pyarrow 25.0.0 for the table scripts). GPU runs used a separate virtual environment whose package versions were not recorded. `resnet18_cifar100_recall_labels/` ran with Python 3.9.

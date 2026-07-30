# Loss spike postmortem — run c898d (smallcnnzoo-mnist pretraining)

*2026-07-07 — run `c898d_00000` (job 24380274, 2026-07-02); quantification jobs 24479666
(`[0,4,8]`) and 24485628 (`[8]`-only).
W&B notes: https://wandb.ai/moos-vu/sane-mnist-smallcnnzoo/runs/c898d_00000*

## What happened

- Train loss spiked 0.086 → 0.98 at epoch 23 (recon R² 0.975 → 0.49), recovered within ~5 epochs.
- The model re-converged in a **rescaled latent regime**: `debug/z_norm` ~3.5 → ~0.75,
  `debug/z_var` ~0.35 → ~0.036 — and never reverted.

## Cause

Optimization instability near the OneCycleLR maximum (peak at epoch 15 with `pct_start=0.3`;
epoch 23 ran at ~0.9× max LR) with **no gradient clipping enabled**.

AMP forensics: the scaler saves no skip log, but AdamW's per-param `step` counter only counts
*applied* steps while the scheduler steps every batch — their difference counts skipped steps
exactly:

| checkpoint | scheduler steps | applied (Adam) steps | skipped |
|---|---|---|---|
| epoch 15 | 4350 | 4350 | 0 |
| epoch 20 | 5800 | 5800 | 0 |
| epoch 25 | 7250 | 7242 | **8** |
| epoch 30 | 8700 | 8692 | 8 |

So epochs 20–25 saw a burst of fp16-overflow gradients: 8 steps skipped (scale halved 2⁸×
total), and the large-but-*finite* gradients in between were applied unclipped. NT-Xent is
cosine-based / scale-free, so nothing pulled the latent scale back — the decoder simply adapted
to the new regime. (Skipped steps also don't pause the scheduler: the LR marched on.)

## Damage quantification

Method (`experiments/smallcnnzoo-mnist/compare_spike_checkpoints.py`): build the `[0,4,8]`
DatasetTokens splits **once** (canonicalization is stochastic — rebuilding per checkpoint would
confound the comparison), encode with pre-spike `checkpoint_000020` and final
`checkpoint_000050`, fit identical ridge heads, compare embedding spectra. Results in
`spike_damage/compare_spike_checkpoints.json`; the `[8]`-only run is
`compare_spike_checkpoints_ep8.py` → `compare_spike_checkpoints_8.json`:

| metric (test split) | ckpt 20 (pre-spike) | ckpt 50 (final) |
|---|---|---|
| `[0,4,8]` per-class recall, mean R² | 0.800 | **0.847** |
| `[0,4,8]` test_acc R² | 0.920 | **0.957** |
| `[0,4,8]` training_iteration R² | 0.672 | 0.672 |
| `[0,4,8]` effective rank (128-dim z) | 14.4 | **5.4** |
| `[8]`-only per-class recall, mean R² | **0.823** | 0.800 |
| `[8]`-only test_acc R² | 0.929 | 0.924 |
| `[8]`-only effective rank | 11.8 | **4.8** |

e550b anchors (10 resplits, `recall_prediction/r2_spread/`): 0.8420 ± 0.0022 on `[0,4,8]`,
0.8407 ± 0.0044 on `[8]`. Embeddings for refits: `spike_damage/embeddings_8.pt`.

Per-class detail on `[8]`: the gap is not a class-1 artifact — class 1 recall is near-ceiling
at epoch 8 (mean 0.967; target variance ~9% of other classes', with 92% of models ≥0.95 and a
3% failure tail carrying 96% of the variance), so its R² is low for *any* encoder — e550b
shows the same drop (0.76 on `[0,4,8]` → 0.31 on `[8]`, `r2_spread.json`). Excluding it the
gap persists (0.891 vs 0.871). Seven of ten classes degrade, dominated by class 5
(0.887 → 0.774); classes 2/4/7/8 each lose 0.02–0.03. The spectra tell the same story:
top-5 eigenvalue share 0.69 → 0.93, and the best ridge penalty jumps 1 → 10 — the collapsed
embedding needs 10× stronger regularization to generalize.

## Conclusions

- **The damage is task-dependent.** On the trajectory-spanning `[0,4,8]` task the final
  checkpoint is strictly better (0.847 vs 0.800, and above the e550b anchor 0.842). But
  restricted to **converged checkpoints only** (`[8]`), the ranking flips: pre-spike 0.823 vs
  final **0.800**, with both below the e550b anchor (0.841). The rank-collapsed final encoder
  resolves coarse trajectory position extremely well but lost fine-grained discrimination
  *among converged models* — which is precisely the population the surgery work edits.
- **Both TODO hypotheses were true simultaneously**: the latent geometry permanently
  concentrated (effective rank 14.4 → 5.4 pooled, 11.8 → 4.8 converged-only; ~92% of variance
  in 5 of 128 directions), and whether that is benign depends on the downstream task's need
  for within-converged-population resolution.
- Checkpoint choice: `checkpoint_000050` for trajectory-wide tasks; for converged-only work
  `checkpoint_000020` is better — but the clean clipped rerun (`gradient-clip-2.0_7a92c`,
  no spike, healthy z_var) should be evaluated next and likely supersedes both.

## Implications for the trust region / surgery work

- The final encoder packs its predictive information into a ~5-dimensional effective subspace.
  The `[8]`-only result gives the first evidence this is costly where it matters: reduced
  resolution among converged models — the population Algorithm 1 edits. Gradients
  backpropagated through the encoder are largely confined to that subspace, which may limit
  the expressible edit directions (can a *class-selective* forgetting direction be represented
  in 5 effective dims?). Suggestively, the resolution loss is itself class-selective — class 5
  recall R² dropped 0.11 while classes 0/6/9 were untouched — i.e. some per-class directions
  survived the collapse and others didn't. The regularizer-that-keeps-edits-faithful
  counter-hypothesis remains untested.
- Natural controlled experiment: run the edit loop with ckpt20 (erank 14.4) vs ckpt50
  (erank 5.4) as the encoder — same run, same data, different geometry — and compare
  steps-until-unfaithfulness and forget/retain endpoints.
- TODO item 4 (latent-scale anchoring: small `training::z_norm_penalty` or pre-decoder
  normalization) remains the structural guard against silent regime re-rolls.

## Prevention (landed for future runs)

- `training::gradient_clipping = "norm"`, `training::gradient_clipp_value = 1.0` in the
  pretrain script (AMP-safe: grads are unscaled before clipping).
- New per-epoch W&B metrics: `debug/skipped_steps` (scale-decrease detection),
  `debug/clipped_steps`, `debug/pre_clip_norm_max`, `debug/pre_clip_norm_median`.
- Val-only monitoring during development (`training::eval_testset = False`).
- First clipped rerun (`926b8`) died at epoch 20 on the home-quota hard limit while writing a
  checkpoint — unrelated to training; pretraining output now goes to
  `/projects/prjs2156/shared/wsl/metanets/sane_pretraining`.

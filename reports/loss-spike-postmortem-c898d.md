# Loss spike postmortem — run c898d (smallcnnzoo-mnist pretraining)

*2026-07-07 — run `c898d_00000` (job 24380274, 2026-07-02); quantification job 24479666.
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
`checkpoint_000050`, fit identical ridge heads, compare embedding spectra. Results
(`spike_damage/compare_spike_checkpoints.json`):

| metric (test split) | ckpt 20 (pre-spike) | ckpt 50 (final) |
|---|---|---|
| per-class recall, mean R² | 0.800 | **0.847** |
| test_acc R² | 0.920 | **0.957** |
| training_iteration R² | 0.672 | 0.672 |
| effective rank (128-dim z) | 14.4 | **5.4** |
| top-5 eigenvalue share | 0.62 | 0.92 |
| z_norm / z_var (sample-level)* | 30.6 / 6.48 | 42.0 / 3.62 |

*Pooled sample embeddings — not comparable to the token-level `debug/z_norm` training metric.

## Conclusions

- **The spike did not hurt downstream quality.** The final checkpoint is strictly better on
  recall/test_acc R², and beats the e550b encoder on the identical `[0,4,8]` task
  (0.8474 vs 0.8420 ± 0.0022 over 10 resplits, from `recall_prediction/r2_spread/`). Use
  `checkpoint_000050` for downstream work.
- **Both TODO hypotheses were true simultaneously**: the latent geometry did permanently
  concentrate (~3× lower effective rank; 92% of variance in 5 of 128 directions), *and* it was
  benign for linear property prediction — 30 extra epochs more than paid for it.

## Implications for the trust region / surgery work

- The final encoder packs its predictive information into a ~5-dimensional effective subspace.
  Ridge regression doesn't care, but weight-space editing might: gradients backpropagated
  through the encoder are largely confined to that subspace, which could limit the expressible
  edit directions (can a *class-selective* forgetting direction be represented in 5 effective
  dims?) or, conversely, act as a regularizer that keeps edits on-manifold and faithful longer.
  Direction unknown — treat as an open empirical question.
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

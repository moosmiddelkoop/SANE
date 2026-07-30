# TODO: loss spike in smallcnnzoo-mnist pretraining (run `c898d`, job 24380274, 2026-07-02)

## Problem

- Train loss spiked 0.086 → 0.98 at epoch 23 (recon R² 0.975 → 0.49), recovered within ~5 epochs.
- The model re-converged in a **rescaled latent regime**: `debug/z_norm` ~3.5 → ~0.75, `debug/z_var` ~0.35 → ~0.036 (defined in `def_loss.py:290-293`: mean embedding L2 norm, and per-dimension batch variance).
- Cause: optimization instability near the OneCycleLR maximum (peak at epoch 15 with `pct_start=0.3`; epoch 23 ran at ~0.9× max LR) with **no gradient clipping enabled**. NT-Xent uses cosine similarity, so latent scale is unconstrained by the loss — after the blow-up nothing pulled the scale back, the decoder just adapted.
- Risk: the 10× drop in `z_var` may indicate partial dimensional collapse, which would hurt downstream property prediction / unlearning work that relies on embedding quality.

## Plan

1. ~~**Quantify the damage**~~ DONE (job 24479666, `experiments/smallcnnzoo-mnist/spike_damage/compare_spike_checkpoints.json`):
   ckpt20 → ckpt50: recall mean R² 0.800 → **0.847**, test_acc R² 0.920 → **0.957**, effective rank 14.4 → **5.4**.
   The spike concentrated the embedding geometry (~3× lower rank) but did **not** hurt property prediction — final checkpoint is strictly better; use it. Open question for the surgery work: does the ~5-dim effective subspace constrain edit directions / the trust region?
2. ~~**Prevent recurrence**~~ DONE: clipping enabled in the pretrain script (norm, 1.0); AMP-skip forensics confirmed 8 skipped steps in epochs 20–25 (0 elsewhere) via AdamW-vs-scheduler step counts. Also added per-epoch `debug/skipped_steps`, `debug/clipped_steps`, `debug/pre_clip_norm_{max,median}` metrics.
3. If spikes persist: lower `optim::lr` slightly or increase warmup (`pct_start`).
4. Consider anchoring the latent scale explicitly (small `training::z_norm_penalty`, or normalizing embeddings before the decoder) so an instability can't silently re-roll the latent geometry mid-run.
5. ~~Record the conclusion in the W&B run notes~~ DONE: https://wandb.ai/moos-vu/sane-mnist-smallcnnzoo/runs/c898d_00000

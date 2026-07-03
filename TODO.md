# TODO: loss spike in smallcnnzoo-mnist pretraining (run `c898d`, job 24380274, 2026-07-02)

## Problem

- Train loss spiked 0.086 → 0.98 at epoch 23 (recon R² 0.975 → 0.49), recovered within ~5 epochs.
- The model re-converged in a **rescaled latent regime**: `debug/z_norm` ~3.5 → ~0.75, `debug/z_var` ~0.35 → ~0.036 (defined in `def_loss.py:290-293`: mean embedding L2 norm, and per-dimension batch variance).
- Cause: optimization instability near the OneCycleLR maximum (peak at epoch 15 with `pct_start=0.3`; epoch 23 ran at ~0.9× max LR) with **no gradient clipping enabled**. NT-Xent uses cosine similarity, so latent scale is unconstrained by the loss — after the blow-up nothing pulled the scale back, the decoder just adapted.
- Risk: the 10× drop in `z_var` may indicate partial dimensional collapse, which would hurt downstream property prediction / unlearning work that relies on embedding quality.

## Plan

1. **Quantify the damage**: encode the same checkpoints with pre-spike `checkpoint_000020` and the final checkpoint; compare downstream property-prediction R² and the eigenspectrum / effective rank of the embeddings.
2. **Prevent recurrence**: set `config["training::gradient_clipping"] = "norm"` (`training::gradient_clipp_value` ~1–5) in the pretrain script — the hook exists in `AEModule` but is off by default.
3. If spikes persist: lower `optim::lr` slightly or increase warmup (`pct_start`).
4. Consider anchoring the latent scale explicitly (small `training::z_norm_penalty`, or normalizing embeddings before the decoder) so an instability can't silently re-roll the latent geometry mid-run.
5. Record the conclusion in the W&B run notes for `c898d_00000`.

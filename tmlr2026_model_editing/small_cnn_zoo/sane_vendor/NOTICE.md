# Vendored SANE code

Files `def_AE.py`, `def_transformer.py`, `def_loss.py` are copied from
https://github.com/HSG-AIML/SANE (commit on main, cloned 2026-10-09), the reference
implementation of Schürholt, Mahoney, Borth, "Towards Scalable and Versatile Weight
Space Learning", ICML 2024 (arXiv:2406.09997). The repository is licensed under the
MIT License (setup.cfg: `license = MIT`; README section "License"). `def_transformer.py`
is itself derived from Karpathy's nanoGPT (MIT).

Local modifications (kept minimal):
- relative import of `def_transformer`;
- `einops.repeat` replaced by `torch.Tensor.expand` (removes the einops dependency);
- `torchmetrics.functional.explained_variance` replaced by an inline computation
  (removes the torchmetrics dependency).

The data pipeline (ray/FFCV, modelzoos.cc checkpoints), training loop and downstream
code are NOT vendored; `sane_zoo.py` re-implements tokenisation, standardisation,
alignment and training for the Unterthiner Small CNN Zoo on top of these modules.

# Val/test contrastive loss discrepancy after the dataloading change (e550b vs c898d)

*2026-07-07 — smallcnnzoo-mnist pretraining*

## Observation

After switching pretraining to the consolidated in-RAM dataset (commit `1e25b01`), val/test
NT-Xent loss got dramatically worse (`c898d`, `926b8`) compared to the per-sample-file run
(`e550b`) — while train loss stayed essentially identical. Suspected leakage or an accidental
regression.

## Root cause: batch composition, not model quality

NT-Xent's difficulty is defined by batch composition — the negatives for each anchor are the
other samples in its batch. The chain:

1. Preprocessing writes per-sample files **model-major**: `sample_N.pt` indices run model by
   model, with epochs 0–8 consecutive within each model (val: 31833 samples = 3537 models × 9).
2. The old `PreprocessedSamplingDataset` listed files with unsorted `os.scandir()` — GPFS
   returns entries in effectively hash order, so the dataset order was *accidentally shuffled*.
3. `data/consolidate_preprocessed.py` uses `sorted(files)` for determinism — which **restored
   the model-major grouping** in the stacked tensors.
4. Val/test loaders use `shuffle=False` (`def_AE_trainable.load_datasets`), so their batch
   composition equals storage order. Train uses `shuffle=True`, so train batches are i.i.d.
   mixtures in all runs — hence no train discrepancy.

Verified empirically: in sorted order, `sample_10000..10007` are one model climbing
training_iteration 1→8 (test_acc 0.46→0.94), then `sample_10008` resets to iteration 0 for the
next model.

Effect: an `e550b` val batch was a random mix (~0.1 expected same-model negatives per anchor);
a `c898d` val batch is ~57 models × 9 consecutive checkpoints, so every anchor faces **8
near-duplicate "negatives"** (its own trajectory neighbors). NT-Xent punishes these false
negatives with high loss even for a perfect encoder.

## Conclusions

- **No leakage, no model regression.** The metric changed meaning; neither number is wrong,
  they measure different tasks. Re-evaluating the e550b checkpoint on the grouped valset would
  look equally "bad".
- Val/test contrastive loss is **not comparable across the consolidation boundary**. Compare
  embedding quality via downstream property-prediction R² instead (e.g. `c898d` final
  checkpoint reaches mean per-class-recall R² 0.847 — see the loss-spike report).
- Proposed fix (not yet applied): wrap val/test sets in a `torch.utils.data.Subset` with one
  fixed seeded permutation in `load_datasets` — restores i.i.d.-like batches, deterministic
  across epochs, no 42 GB data rewrite. Runs after the fix will again be loss-comparable to
  `e550b` (and to each other), not to `c898d`/`926b8`.

## Trust-region brainstorm this triggered

Question: for the weight-space surgery goal (CNNZoo_surgery paper §4.3: the edit loop must stop
before ψ "leaves its trust region" — predicted recall diverging from actual recall of the
edited model), would same-model different-epoch **in-batch negatives** help learn a more
expressive space, or is **permutation augmentation** the better lever?

Insights:

- The two act on different axes: permutation/noise augmentations define what counts as *same*
  (positives); batch composition defines what counts as *different* (negatives).
- **Permutation positives are load-bearing for the trust region.** They align embedding
  geometry with *function* (the whole permutation orbit maps together), and faithfulness is a
  statement about function. Noise augmentation is even more directly trust-region shaped: it
  teaches z(θ+ε) ≈ z(θ), i.e. smoothness exactly in the small-off-manifold-step regime where
  Algorithm 1 operates.
- **Temporal hard negatives are double-edged.** Distant epochs (0 vs 8) are genuinely different
  functions — fine as negatives, and finer trajectory resolution could sharpen the "reverse the
  training of class c" edit direction. But adjacent late epochs (7 vs 8) are functionally
  near-identical: as false negatives they force the encoder to amplify tiny non-functional
  weight differences (SGD noise), making predictions brittle along edit trajectories — the
  exact §4.3 failure mode, arriving sooner.
- Middle ground if trajectory resolution is wanted: model-grouped batches over a *spread* epoch
  subset (e.g. {0,4,8}) so temporal negatives are always functionally distant, or
  supervised-contrastive treatment of near-adjacent epochs as weak positives.
- **Evaluate on the metric that matters**, not val contrastive loss (batch-composition
  relative): run the edit loop on held-out models and measure steps-until-unfaithfulness
  (|predicted − actual recall| under tolerance along the edit path) plus forget/retain
  endpoints.
- SANE framing: the pipeline is encoder → frozen linear ridge head. The head is globally
  well-behaved, so the trust region lives or dies with the **encoder's extrapolation** — which
  makes pretraining augmentation/batch choices the main lever on §4.3 (invisible in the paper's
  raw-weight MLP version).

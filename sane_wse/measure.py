"""Does canonicalization matter for SANE's per-class predictions? Measure it, on held-out models.

The AE and the recall head were trained with `map_to_canonical=True`, but the exported meta-network
feeds the encoder un-aligned weights. This scores the lens against the `wse` indexer's ground-truth
per-class recall, once per manifest, so the answer comes from data instead of from an argument.

This script imports `wse` to read the zoo and its index. The dependency only runs this way: `wse`
never imports SANE, which is why the lens lives here.

Run from the SANE repo root, with wse and SANE importable:
    python sane_wse_canonicalization.py <indexed_zoo> <out.json> <manifest> [<manifest> ...]
"""

import json
import sys
import time
from pathlib import Path

import torch
from wse.zoo import Zoo

from sane_wse import from_manifest

DEVICE = "cpu"  # weight matching is scipy on the host, and the encoder is fast enough here
BATCH_SIZE = 32


def score(lens, batches, n_classes):
    """Predicted and true per-class accuracy for every model, plus seconds per model."""
    predicted, truth, model_ids, seconds = [], [], [], 0.0
    for batch in batches:
        start = time.time()
        with torch.no_grad():
            predicted.append(lens(batch.params.params))
        seconds += time.time() - start
        truth.append(torch.tensor(batch.ground_truth[[f"acc_class_{index}" for index in range(n_classes)]].values))
        model_ids.extend(batch.model_ids)
    predicted, truth = torch.cat(predicted).double(), torch.cat(truth).double()
    return predicted, truth, model_ids, seconds / len(predicted)


def report(predicted, truth):
    """Per-class R^2 and mean absolute error, over the held-out models."""
    residual = ((truth - predicted) ** 2).sum(dim=0)
    total = ((truth - truth.mean(dim=0)) ** 2).sum(dim=0)
    r2 = 1 - residual / total
    return {
        "r2_per_class": [round(value, 4) for value in r2.tolist()],
        "mean_r2": round(float(r2.mean()), 4),
        "mae": round(float((truth - predicted).abs().mean()), 4),
        "negative_prediction_rate": round(float((predicted < 0).double().mean()), 4),
        "models_with_a_negative_prediction": int((predicted < 0).any(dim=1).sum()),
    }


def compare(first, second, samples=2000):
    """Paired comparison of two runs over the same models: does the difference survive resampling?

    Aggregate R^2 can be unchanged while individual predictions move, and unlearning is driven by
    individual predictions, so both are reported.
    """
    truth = torch.tensor(first["truth"])
    a, b = torch.tensor(first["predicted"]), torch.tensor(second["predicted"])

    def mean_r2(predicted, index):
        selected, expected = predicted[index], truth[index]
        residual = ((expected - selected) ** 2).sum(dim=0)
        total = ((expected - expected.mean(dim=0)) ** 2).sum(dim=0)
        return float((1 - residual / total).mean())

    generator = torch.Generator().manual_seed(0)
    all_models = torch.arange(len(truth))
    resampled = [
        mean_r2(b, index) - mean_r2(a, index)
        for index in (torch.randint(0, len(truth), (len(truth),), generator=generator) for _ in range(samples))
    ]
    resampled = torch.tensor(resampled)
    delta = (a - b).abs()
    return {
        "mean_r2_difference": round(mean_r2(b, all_models) - mean_r2(a, all_models), 4),
        "bootstrap_95_ci": [round(float(resampled.quantile(0.025)), 4), round(float(resampled.quantile(0.975)), 4)],
        "per_prediction_absolute_difference": {
            "mean": round(float(delta.mean()), 4),
            "median": round(float(delta.median()), 4),
            "max": round(float(delta.max()), 4),
        },
    }


def main():
    zoo_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    manifests = [Path(argument) for argument in sys.argv[3:]]

    zoo = Zoo.open(zoo_dir)
    selection = zoo.select()
    print(f"{len(selection)} held-out models from {zoo_dir}")

    results = {}
    for manifest in manifests:
        lens = from_manifest(manifest)
        lens.prepare(zoo.reference_shapes(), torch.device(DEVICE))
        lens.eval().requires_grad_(False)

        batches = selection.batches(BATCH_SIZE, device=DEVICE)
        predicted, truth, model_ids, seconds = score(lens, batches, zoo.n_classes)
        results[manifest.stem] = report(predicted, truth) | {
            "canonicalized": lens.permutation_spec is not None,
            "n_models": len(predicted),
            "seconds_per_model": round(seconds, 4),
            "predicted": predicted.tolist(),
            "truth": truth.tolist(),
        }
        results[manifest.stem]["model_ids"] = model_ids
        summary = {key: value for key, value in results[manifest.stem].items() if key not in ("predicted", "truth", "model_ids")}
        print(f"{manifest.stem}: {json.dumps(summary)}")

    if len(manifests) == 2:
        results["comparison"] = compare(*results.values())

    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

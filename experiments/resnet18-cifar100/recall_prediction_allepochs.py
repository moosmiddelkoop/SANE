"""Per-class recall heads on SANE embeddings of every 5th checkpoint of the CIFAR-100 ResNet-18 zoo.

--encoder sane (default): the SANE run pretrained on all checkpoints (epochs 0, 5, ..., 60; tk288).
Each checkpoint is encoded as in property_prediction_cifar100_resnet18_mlp.py: consecutive
windows of the training windowsize, mean over all token embeddings. The zoo is loaded in the
same 28 model chunks as the preprocessing (data/preprocess_dataset_resnet18_cifar100_tk288_allepochs.py),
so each chunk gets the same standardization statistics as in pretraining. Embeddings and
targets of all 13 epochs are cached in recall_prediction/allepochs/embeddings_<run>.pt; if the
cache exists, encoding is skipped.

--encoder lq replaces SANE with LayerQuintiles statistics of the raw weights, as a control:
per layer (weight and bias of each conv and fc layer, no batch norm; 21 layers) the mean,
variance and the 1/25/50/75/99th percentiles, read from the checkpoints directly (no
canonicalization, standardization or token padding). Cached in embeddings_lq_raw.pt.

The heads use epochs 5..60 only (epoch 0 is the random init). Two scopes:
- general: one head fit on the checkpoints of all these epochs, scored per epoch
- per_epoch: one head per epoch, fit and scored on that epoch only
Two heads, both on the zoo's split.json:
- ridge (closed form): features standardized with the train statistics, fit on train,
  regularizer chosen on val
- MLP (eval_per_class_recall_MLP): fit on train, early stopping on val, one run per seed.
  Features and targets are standardized with the train statistics, else the MLP only
  learns the class means; predictions are mapped back to recall.
Reference: the class-mean predictor (train mean of each class at that epoch).

Metrics per epoch on train and test, in recall units: MSE, MAE, R^2 per class over the
models (mean over classes), R^2 per model over the classes (mean over models), and the
interaction R^2: one R^2 over all (model, class) pairs of the double-centered targets and
predictions (each model's mean and each class's mean removed), so any prediction of the
form model term + class term, such as the class-mean predictor, scores exactly 0.
Results go to recall_prediction/allepochs/per_class_recall_<run>.json; plot them with
plot_recall_prediction_allepochs.py. Run with the sibling .sh:
sbatch recall_prediction_allepochs.sh [sane|lq] (CPU only for lq or with a cache:
sbatch --partition=rome --gpus=0 recall_prediction_allepochs.sh lq).
"""

import logging
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import gc
import json
import statistics
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from SANE.datasets.dataset_tokens import DatasetTokens
from SANE.datasets.zoo_split import assert_same_split, load_split
from SANE.git_re_basin.git_re_basin import resnet18_permutation_spec
from SANE.models.def_AE_module import AEModule
from SANE.models.def_downstream_module import DownstreamTaskLearner
from SANE.utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = ArgumentParser()
parser.add_argument("--encoder", choices=["sane", "lq"], default="sane")
ENCODER = parser.parse_args().encoder

RUN = "cifar100-resnet18-tk288-ep0-60-v1.0_d75a9_00000_2026-10-28_12-08-02"
CHECKPOINT = Path("/projects/prjs2156/shared/wsl/metanets/sane_pretraining/cifar100_resnet18") / RUN / "checkpoint_000050"
ZOO_ROOT = Path("/projects/prjs2156/shared/wsl/cifar100_resnet18/tune_zoo_cifar100_resnet18_kaiming_uniform")
OUT_DIR = Path("recall_prediction/allepochs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TAG = RUN if ENCODER == "sane" else "lq_raw"
EMBEDDINGS_PT = OUT_DIR / f"embeddings_{TAG}.pt"
RESULTS_JSON = OUT_DIR / f"per_class_recall_{TAG}.json"

ENCODED_EPOCHS = list(range(0, 61, 5))  # as in the preprocessing
HEAD_EPOCHS = list(range(5, 61, 5))
N_CHUNKS = 28  # as in the preprocessing
ACC_CLASS_KEYS = [f"acc_class_{i}" for i in range(100)]
QUANTILES = torch.tensor([0.01, 0.25, 0.5, 0.75, 0.99])  # as in LayerQuintiles
SPLITS = ["train", "val", "test"]

SEED = 42
MLP_SEEDS = [0, 1, 2]
MLP_EPOCHS = 200
LR = 1e-3
MLP_BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
seed_everything(SEED)
dstk = DownstreamTaskLearner()
dstk.polar_coordinates = False
dstk.device = torch.device(device)


def embed_models(tokens, positions, module, windowsize):
    """Mean token embedding per model: tokens [N, T, tokensize] -> [N, lat_dim].

    The sequence is cut into consecutive windows of the training windowsize (no halo).
    The last window is moved back so that it is full; the tokens it repeats from the
    previous window are dropped, so every token counts once in the mean."""
    sequence_length = tokens.shape[1]
    starts = [min(start, sequence_length - windowsize) for start in range(0, sequence_length, windowsize)]
    tokens_repeated_by_last_window = starts[-2] + windowsize - starts[-1]
    window_positions = torch.stack([positions[s : s + windowsize] for s in starts]).int().to(device)
    embeddings = []
    for model_tokens in tokens:
        windows = torch.stack([model_tokens[s : s + windowsize] for s in starts]).to(device)
        with torch.no_grad(), torch.autocast("cuda"):
            z = module.forward_encoder(windows, window_positions)  # [n_windows, windowsize, lat_dim]
        z = torch.cat([z[:-1].flatten(0, 1), z[-1, tokens_repeated_by_last_window:]])
        assert z.shape[0] == sequence_length
        embeddings.append(z.float().mean(dim=0).cpu())
    return torch.stack(embeddings)


def layer_quintile_features(checkpoint_path):
    """LayerQuintiles statistics of the raw weights: 7 per conv / fc layer (weight and bias)."""
    state = torch.load(checkpoint_path, map_location="cpu")
    features = []
    for name, weight in state.items():
        if name.endswith(".weight") and weight.dim() > 1:  # batch-norm weights are 1-D
            bias = state.get(name.removesuffix("weight") + "bias", torch.empty(0))
            values = torch.cat([weight.flatten(), bias.flatten()]).float()
            features += [values.mean(), values.var(unbiased=False), *torch.quantile(values, QUANTILES)]
    return torch.stack(features)


def recall_targets(model_dir):
    lines = [json.loads(line) for line in (model_dir / "result.json").read_text().splitlines()]
    line_of_epoch = {line["training_iteration"]: line for line in lines}
    return torch.tensor([[line_of_epoch[epoch][key] for key in ACC_CLASS_KEYS] for epoch in ENCODED_EPOCHS])


if EMBEDDINGS_PT.exists():
    logging.info(f"Loading cached embeddings from {EMBEDDINGS_PT}")
    cache = torch.load(EMBEDDINGS_PT, map_location="cpu")
elif ENCODER == "sane":
    logging.info(f"Loading pretrained SANE model from {CHECKPOINT}")
    config = json.load((CHECKPOINT.parent / "params.json").open("r"))
    config["device"] = device
    config["model::compile"] = False
    config["training::steps_per_epoch"] = 1
    module = AEModule(config)
    checkpoint = torch.load(CHECKPOINT / "state.pt", map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model"].items()}
    module.model.load_state_dict(state_dict)
    module.model.eval()

    z_all, Y_all, model_dirs, model_split = [], [], [], []
    for split in SPLITS:
        for chunk_index in range(N_CHUNKS):
            logging.info(f"Building DatasetTokens split={split}, chunk {chunk_index + 1} of {N_CHUNKS}")
            # must match the preprocessing (data/preprocess_dataset_resnet18_cifar100_tk288_allepochs.py)
            ds = DatasetTokens(
                root=ZOO_ROOT,
                epoch_lst=ENCODED_EPOCHS,
                mode="vector",
                permutation_spec=resnet18_permutation_spec(),
                map_to_canonical=True,
                standardize=True,
                tokensize=config["ae:i_dim"],
                train_val_test=split,
                model_chunk=(chunk_index, N_CHUNKS),
                weight_threshold=float("inf"),
                max_samples=None,
                property_keys={"result_keys": ACC_CLASS_KEYS, "config_keys": []},
                num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)),
                verbosity=3,
                getitem="tokens+props",
                ignore_bn=True,
            )
            assert_same_split(config, ds)
            for model_index, model_epochs in enumerate(ds.epochs):
                assert model_epochs == ENCODED_EPOCHS, f"{ds.paths[model_index][0]} has epochs {model_epochs}"
                tokens = torch.stack(ds.data[model_index])  # [epochs, T, tokensize]
                z_all.append(embed_models(tokens, ds.positions, module, config["training::windowsize"]))
                Y_all.append(torch.tensor([[ds.properties[key][model_index][e] for key in ACC_CLASS_KEYS] for e in range(len(ENCODED_EPOCHS))]))
                model_dirs.append(Path(ds.paths[model_index][0]).name)
                model_split.append(split)
            split_id = ds.split_id
            del ds
            gc.collect()

    cache = {
        "z": torch.stack(z_all),  # [models, epochs, lat_dim]
        "Y": torch.stack(Y_all),  # [models, epochs, classes]
        "epochs": ENCODED_EPOCHS,
        "model_dirs": model_dirs,  # zoo dir name per row
        "model_split": model_split,  # split.json split per row
        "split_id": split_id,
        "encoder": str(CHECKPOINT),
    }
    torch.save(cache, EMBEDDINGS_PT)
    logging.info(f"Saved embeddings cache to {EMBEDDINGS_PT}")
else:
    split = load_split(ZOO_ROOT)
    model_dirs = [d for s in SPLITS for d in split.paths(s)]
    checkpoint_paths = [d / f"checkpoint_{epoch:06d}" / "checkpoints" for d in model_dirs for epoch in ENCODED_EPOCHS]
    logging.info(f"LayerQuintiles of {len(checkpoint_paths)} checkpoints")
    with ThreadPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as pool:
        features = torch.stack(list(pool.map(layer_quintile_features, checkpoint_paths)))
    cache = {
        "z": features.view(len(model_dirs), len(ENCODED_EPOCHS), -1),  # [models, epochs, 147]
        "Y": torch.stack([recall_targets(d) for d in model_dirs]),  # [models, epochs, classes]
        "epochs": ENCODED_EPOCHS,
        "model_dirs": [d.name for d in model_dirs],
        "model_split": [s for s in SPLITS for _ in split.paths(s)],
        "split_id": split.split_id,
        "encoder": "LayerQuintiles of the raw weights",
    }
    torch.save(cache, EMBEDDINGS_PT)
    logging.info(f"Saved embeddings cache to {EMBEDDINGS_PT}")

# a cache from another split could put test models into the heads' training data
assert cache["split_id"] == load_split(ZOO_ROOT).split_id, f"{EMBEDDINGS_PT} is from another split; delete it"
head_epoch_index = [cache["epochs"].index(epoch) for epoch in HEAD_EPOCHS]
rows = {split: torch.tensor([s == split for s in cache["model_split"]]) for split in SPLITS}
z = {split: cache["z"][rows[split]][:, head_epoch_index].float() for split in SPLITS}  # [models, epochs, lat_dim]
Y = {split: cache["Y"][rows[split]][:, head_epoch_index].double() for split in SPLITS}  # [models, epochs, classes]
logging.info(", ".join(f"{s}: {len(z[s])} models" for s in SPLITS))


def double_center(matrix):
    """[models, classes] -> the model x class interaction: remove the model means and the class means"""
    return matrix - matrix.mean(dim=1, keepdim=True) - matrix.mean(dim=0) + matrix.mean()


def recall_metrics(predictions, targets):
    """predictions and targets [models, classes] in recall units."""
    errors = predictions.double() - targets
    interaction = double_center(targets)
    r2_interaction = 1 - ((interaction - double_center(predictions.double())) ** 2).sum() / (interaction**2).sum()
    squared_errors = errors**2
    r2_per_class = 1 - squared_errors.sum(dim=0) / ((targets - targets.mean(dim=0)) ** 2).sum(dim=0)
    r2_per_model = 1 - squared_errors.sum(dim=1) / ((targets - targets.mean(dim=1, keepdim=True)) ** 2).sum(dim=1)
    return {
        "mse": squared_errors.mean().item(),
        "mae": errors.abs().mean().item(),
        "r2_per_class": r2_per_class.mean().item(),
        "r2_per_model": r2_per_model.mean().item(),
        "r2_interaction": r2_interaction.item(),
    }


def fit_ridge(z_fit, Y_fit):
    """z_fit {split: [rows, lat_dim]}, Y_fit {split: [rows, classes]} -> predict function, info"""
    # one ridge penalty for all features, so put them on one scale (LayerQuintiles features are not)
    z_mean, z_std = z_fit["train"].mean(dim=0), z_fit["train"].std(dim=0)
    z_fit = {s: (z_fit[s] - z_mean) / z_std for s in SPLITS}
    ridge = dstk.compute_closed_form_solution_multivariate(
        z_train=z_fit["train"], Y_train=Y_fit["train"], z_test=z_fit["test"], Y_test=Y_fit["test"],
        z_val=z_fit["val"], Y_val=Y_fit["val"],
    )
    B = ridge["B"].cpu().double()
    predict = lambda z_split: torch.cat([(z_split - z_mean) / z_std, torch.ones(len(z_split), 1)], dim=1).double() @ B
    return predict, {"reg_best": ridge["reg_best"]}


def fit_mlp(z_fit, Y_fit):
    z_mean, z_std = z_fit["train"].mean(dim=0), z_fit["train"].std(dim=0)
    Y_mean, Y_std = Y_fit["train"].mean(dim=0), Y_fit["train"].std(dim=0)
    standardized = {s: (((z_fit[s] - z_mean) / z_std).float(), ((Y_fit[s] - Y_mean) / Y_std).float()) for s in SPLITS}
    result = dstk.eval_per_class_recall_MLP(
        model=None,
        trainset=standardized["train"],
        valset=standardized["val"],
        testset=standardized["test"],
        target_keys=ACC_CLASS_KEYS,
        epochs=MLP_EPOCHS,
        lr=LR,
        mlp_batch_size=MLP_BATCH_SIZE,
    )

    def predict(z_split):
        with torch.no_grad():
            return result["mlp"](((z_split - z_mean) / z_std).to(dstk.device)).cpu().double() * Y_std + Y_mean

    return predict, {"best_epoch": result["best_epoch"]}


def score_per_epoch(predict, epoch_indices):
    return {
        HEAD_EPOCHS[e]: {s: recall_metrics(predict(z[s][:, e]), Y[s][:, e]) for s in ["train", "test"]}
        for e in epoch_indices
    }


def run_head(fit):
    """Both scopes for one head; returns {"general": {epoch: {split: metrics}}, "per_epoch": ...}"""
    all_epochs = range(len(HEAD_EPOCHS))
    predict, info = fit({s: z[s].flatten(0, 1) for s in SPLITS}, {s: Y[s].flatten(0, 1) for s in SPLITS})
    general = {"info": info, "per_epoch": score_per_epoch(predict, all_epochs)}
    per_epoch = {"info": {}, "per_epoch": {}}
    for e in all_epochs:
        predict, info = fit({s: z[s][:, e] for s in SPLITS}, {s: Y[s][:, e] for s in SPLITS})
        per_epoch["info"][HEAD_EPOCHS[e]] = info
        per_epoch["per_epoch"].update(score_per_epoch(predict, [e]))
    return {"general": general, "per_epoch": per_epoch}


def mean_and_std_over_seeds(runs):
    """runs: list of run_head outputs -> same nesting with {"mean", "std"} per metric"""
    out = {}
    for scope in ["general", "per_epoch"]:
        out[scope] = {"info": [r[scope]["info"] for r in runs], "per_epoch": {}}
        for epoch in HEAD_EPOCHS:
            out[scope]["per_epoch"][epoch] = {
                s: {
                    metric: {
                        "mean": statistics.mean(r[scope]["per_epoch"][epoch][s][metric] for r in runs),
                        "std": statistics.stdev(r[scope]["per_epoch"][epoch][s][metric] for r in runs),
                    }
                    for metric in runs[0][scope]["per_epoch"][epoch][s]
                }
                for s in ["train", "test"]
            }
    return out


results = {"ridge": run_head(fit_ridge)}
mlp_runs = []
for seed in MLP_SEEDS:
    seed_everything(seed)
    mlp_runs.append(run_head(fit_mlp))
results["mlp"] = mean_and_std_over_seeds(mlp_runs)
results["class_mean"] = {
    epoch: {s: recall_metrics(Y["train"][:, e].mean(dim=0).expand_as(Y[s][:, e]), Y[s][:, e]) for s in ["train", "test"]}
    for e, epoch in enumerate(HEAD_EPOCHS)
}
for epoch in HEAD_EPOCHS:
    test = lambda head, scope: results[head][scope]["per_epoch"][epoch]["test"]
    logging.info(
        f"epoch {epoch:2d} test R^2 per class / per model: "
        f"ridge general {test('ridge', 'general')['r2_per_class']:.3f} / {test('ridge', 'general')['r2_per_model']:.3f}, "
        f"ridge per epoch {test('ridge', 'per_epoch')['r2_per_class']:.3f} / {test('ridge', 'per_epoch')['r2_per_model']:.3f}, "
        f"MLP general {test('mlp', 'general')['r2_per_class']['mean']:.3f} / {test('mlp', 'general')['r2_per_model']['mean']:.3f}, "
        f"MLP per epoch {test('mlp', 'per_epoch')['r2_per_class']['mean']:.3f} / {test('mlp', 'per_epoch')['r2_per_model']['mean']:.3f}, "
        f"class mean {results['class_mean'][epoch]['test']['r2_per_class']:.3f} / {results['class_mean'][epoch]['test']['r2_per_model']:.3f}"
    )

summary = {
    "encoder": cache["encoder"],
    "embeddings_source": str(EMBEDDINGS_PT),
    "split_id": cache["split_id"],
    "head_epochs": HEAD_EPOCHS,
    "mlp": {"epochs": MLP_EPOCHS, "lr": LR, "mlp_batch_size": MLP_BATCH_SIZE, "seeds": MLP_SEEDS},
    "results": results,
}
RESULTS_JSON.write_text(json.dumps(summary, indent=4))
logging.info(f"Wrote results to {RESULTS_JSON}")

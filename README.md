![Overview of the SANE approach](assets/approach_overview.png)

# SANE: Scalable and Versatile Weight Space Learning

This repository contains the code for the paper "Towards Scalable and Versatile Weight Space Learning," presented at ICML 2024. This work introduces SANE*, a novel approach for learning task-agnostic representations of neural networks that are scalable to larger models and applicable to various tasks. The paper can be found here: [ICML proceedings](https://proceedings.mlr.press/v235/schurholt24a.html) | [arxiv](http://arxiv.org/abs/2406.09997).

<sup>*Sequential Autoencoder for Neural Embeddings</sub>

## Summary
Learning representations of well-trained neural network models holds the promise to provide an understanding of the inner workings of those models. However, previous work has faced limitations when processing larger networks or was task-specific to either discriminative or generative tasks. This paper introduces the SANE approach to weight-space learning. SANE overcomes previous limitations by learning task-agnostic representations of neural networks that are scalable to larger models of varying architectures and show capabilities beyond a single task. Our method extends the idea of hyper-representations towards sequential processing of subsets of neural network weights, allowing one to embed larger neural networks as a set of tokens into the learned representation space. SANE reveals global model information from layer-wise embeddings and can sequentially generate unseen neural network models, which was unattainable with previous hyper-representation learning methods. As the figure below shows, extensive empirical evaluation demonstrates that SANE (light blue) matches or exceeds state-of-the-art performance on several weight representation learning benchmarks, particularly in initialization for new tasks and larger ResNet architectures. 

![Performance overview of SANE](assets/radar.png)

### Key Methods

- **Sequential Decomposition**: Breaking down neural network weights into smaller, manageable token sequences.
- **Self-Supervised Pretraining**: Using a self-supervised approach to pretrain the SANE model on a variety of tasks and architectures on **subsequences of models**.
- **Model Analysis**: Analysing models by their embedding sequences.
- **Model Sampling**: Generating new neural network models by sampling from the learned representation space.

### Results

- **Model Property Prediction**: SANE embeddings demonstrate high predictive performance for model properties such as test accuracy, epoch, and generalization gap across various datasets and architectures.
- **Generative Capabilities**: SANE can generate high-performing neural network models from scratch or fine-tune them with significantly less computational effort compared to training from scratch.
- **Scalability**: The method scales to large models like ResNet-18, preserving meaningful information across long sequences of tokens.

## Code Structure

- **data/**: Scripts for data preprocessing and loading.
- **experiments/**: One directory per model zoo (`resnet18-cifar100/`, `cnn-cifar10/`, `smallcnnzoo-{mnist,fmnist,cifar10,svhn}/`) with scripts to pre-train SANE, predict properties and sample models. Scripts that run long have a SLURM batch file (`.sh`) with the same name.
- **src/**: contains the SANE package to preprocess model checkpoint datasets, pre-train SANE, and perform discriminative and generative downstream tasks.
- **reports/**: Short write-ups of training issues found during our runs (a loss spike, a val-loss shift).

## Running Experiments
We include code to run example experiments and showcase how to use our code. 

### Download Model Zoo Datasets
We have made several model zoos available at [modelzoos.cc](https://modelzoos.cc/). Any of these zoos can be used in our pipeline, with minor adjustments.  

To get started with a small experiment, navigate to `./data/` and run 
```bash
bash download_cifar10_cnn_sample.sh
```
This will download and unzip a small model zoo example with CNN models trained on CIFAR-10. 
Before anything else, give the zoo its fixed train/val/test split (see [Create the zoo's split first](#create-the-zoos-split-first)):
```bash
uv run create_split.py <zoo dir>
```
Training on large model zoos requires preprocessing for training efficiency. We provide code to preprocess training samples. To compile those datasets, run
```bash
python3 preprocess_dataset_cnn_cifar10_sample.py
```
in `./data/`. in the same directory, we provide download and preprocessing scripts for other zoos as well. 
The preprocessed datasets have no specific dependency requirements, other than regular numpy and pytorch.

Please note that this is not the exact models used in the paper and will therefore produce different results. The full zoos can be downloaded from [modelzoos.cc](https://modelzoos.cc/) and used in the same way as the zoo sample.  

### Create the zoo's split first
The first thing you do with a new model zoo, before preprocessing, pretraining or any downstream task, is fix its train/val/test split. Do this once per zoo:
```bash
uv run data/create_split.py <zoo dir>
```
This writes `<zoo dir>/split.json`: a seeded shuffle of the model directories, cut 70/15/15 (change with `--ratios 0.8 0.1 0.1` and `--seed`). Every later step reads its models from this file, so train, val and test can never mix, whatever script or config you use. There is no split setting anywhere else.

- **No split, no data.** Without `split.json`, every dataset class stops with `MissingSplitError` and prints the command above.
- **Created once, never changed.** `create_split.py` refuses to overwrite an existing split. Loading fails if the file was edited by hand, or if model directories were added to or removed from the zoo.
- **Checked all the way through.** Each split has a `split_id`. Preprocessing stores it with the dataset, pretraining checks that train/val/test share it and do not overlap, and downstream scripts check that the encoder was pretrained on the same split.
- **Starting over** means deleting `split.json` by hand. Every dataset and encoder built on the old split then stops loading, on purpose: re-run preprocessing and pretraining.
- **Cross-validation** may re-split train and val (`SANE.datasets.zoo_split.resplit_train_val`). The test split never moves.
- **Data made before `split.json` existed** has no `split_id` and fails these checks. To use it anyway, knowing its splits may overlap, set `config["dataset::legacy_unverified_split"] = True`.

### Preprocessing model zoos
Preprocessing turns a zoo of raw checkpoints into the tokenized dataset that pretraining reads. The consolidated pipeline (`SANE.datasets.dataset_preprocessing_consolidated`) runs, per split: discover model directories in the zoo → load the checkpoints listed in `epoch_list` in parallel via Ray → map permutation symmetries to a canonical form with git-re-basin (`map_to_canonical`) → standardize weights per layer → tokenize each checkpoint into `windowsize` tokens of size `tokensize` → stack everything into in-RAM tensors, saved as a single `<out_dir>/dataset.pt` plus `dataset_info_<split>.json` and `dataset_normalization_<split>.json`.

For the small CNN zoos (3 conv layers + dense head), the entry point is
```bash
python3 preprocess_dataset_smallcnnzoo.py --in_dir=<zoo dir> --out_dir=<target dir>
```
in `./data/`. All datasets of this zoo family (MNIST, Fashion-MNIST, CIFAR-10, SVHN) share the same architecture and hence the same token geometry (`tokensize=145`, `windowsize=58`, epochs 0–8), so one script serves them all — `data/preprocess_all_unthi.sh` is a SLURM batch script that runs them back to back (about 5 hours in total). The raw zoo layout is described in `CWI-README.md`.

> **Note — memory.** The consolidated pipeline holds an entire split in RAM: all raw checkpoints as a graph of small Python/tensor objects (~4–5 GB for a 30k-model small-CNN zoo) plus the preallocated output tensors. The final stacking step iterates with a `DataLoader` whose forked workers copy-on-write duplicate that object graph, so peak memory scales with *workers × zoo size*; stacking workers are therefore capped at 8 regardless of `num_threads`. Budget roughly `parent process + 8 × object-graph size` (~50 GB for a 30k-model zoo) when sizing a SLURM allocation.

> **Note — why one `dataset.pt` ("consolidation").** The original pipeline (`SANE.datasets.dataset_preprocessing`) wrote one `.pt` file per sample and read every file again in each training epoch. For a 30k-model zoo that is 200k+ files, and file reads limited pretraining to about one epoch per hour. The consolidated pipeline stores all samples as stacked tensors (`TensorSamplingDataset`) in one file, which pretraining loads into RAM once. The original pipeline is still available. `data/consolidate_preprocessed.py` converts a zoo that was preprocessed with it.
>
> The change also changed the order of the val/test samples: all checkpoints of one model are now next to each other. Val/test batches are not shuffled, so each batch holds near-copies of the same model, and the contrastive val/test loss is much higher. Model quality did not change. Do not compare contrastive val/test loss between runs from before and after the change; compare downstream results instead. Details: `reports/contrastive-val-loss-discrepancy.md`.

### Pretraining SANE
Code to pretrain SANE on the ResNet-18 zoo is contained in `experiments/resnet18-cifar100/pretrain_sane_cifar100_resnet18.py`. The code relies on ray.tune to manage resources, but currently only runs a single config. 
To vary any of the configurations, exchange the value with `tune.grid_search([value_1, ..., value_n])`. To run the experiment, run
```
python3 pretrain_sane_cifar100_resnet18.py
```
in `experiments/resnet18-cifar100/`.

For the small CNN zoos, submit `experiments/smallcnnzoo-<zoo>/pretrain_sane_<zoo>_smallcnnzoo.sh` with `sbatch`. These runs use gradient clipping and log to Weights & Biases.

> **Note — seeds.** Each entry script sets one seed with `seed_everything(SEED)` from `SANE.utils` and passes the same value as `config["seed"]`. The library itself sets no fixed seeds, so the train/val/test split changes with this seed.

### Using SANE embeddings to predict properties
SANE embeddings preserve the sequential decomposition of models. This enables a more fine-granual analysis of models compared to global model embeddings. The figure below shows a comparison between SANE embeddings (right) and features used in the WeightWatcher library (left), which are based on the eigendecomposition of the weight matrices. Both show similar trends of layer properties in ResNet models, but SANE appears to pick up on additional signals in the middle layers.
![Analysis of models: comparing weight matrix eigendecomposition features (left) to SANE embeddings (right)](assets/analysis_layers.png)

We provide code to use SANE embeddings to predict such model properties in `experiments/resnet18-cifar100/property_prediction_cifar100_resnet18.py`. It assumes downloaded dataset and pre-trained SANE as described above. Within `property_prediction_cifar100_resnet18.py`, set the path to the pretrained SANE model and epoch. then run
```bash
python3 property_prediction_cifar100_resnet18.py
```
within `experiments/resnet18-cifar100/`. This will compute the property prediction results from both SANE embeddings and weight-statistic baselines and save them in a `json`.

> **Note — downstream encoding must match pretraining preprocessing.** The property-prediction and sampling/finetune scripts re-tokenize raw checkpoints before passing them through the trained SANE encoder. The `permutation_spec`, `map_to_canonical`, `ignore_bn`, and standardization settings used here **must** match what was used when the pretraining dataset was generated (see `data/preprocess_dataset_*.py`); otherwise embeddings will be off-distribution and downstream metrics will silently degrade. The `epoch_list` is the only preprocessing knob that is free to differ — it just picks which checkpoints to encode.

### Predicting per-class recall (small CNN zoos)
For our unlearning work we predict the recall of each of the 10 classes (`acc_class_0` … `acc_class_9`) from a model's SANE embedding. Each `experiments/smallcnnzoo-<zoo>/` directory has `property_prediction_<zoo>_smallcnnzoo_mlp.py`, submitted via its `.sh`. The script:
1. encodes the final checkpoint (epoch 8) of every zoo model, and caches the embeddings in `recall_prediction/mlp/embeddings_sorted.pt`,
2. splits the models by sorted name: the first 80 % train, the last 20 % test,
3. trains a small MLP head (two hidden layers of 128 units) once per seed,
4. writes MSE, MAE and R² per seed, plus mean and standard deviation, to a JSON file.

A second run re-uses the cached embeddings and only trains the heads. `experiments/smallcnnzoo-mnist/` also contains linear-head variants and analyses of how much the results vary between splits.

### Generating Models
Generating models can provide initializations even for new tasks and architectures that give an advantage over random initializations, see the Figure below.
![SANE weight generation comparison. Models initialized with SANE outperform random initialization on new models or tasks](assets/sample_models.png)

In `experiments`, there is also code to generate and evaluate models. 
`cnn-cifar10_exploration.ipynb` is a quick-start notebook to explore the datasets, SANE models, training loop, as well as encoding and de-coding models. 
We further provide experiment code for the sample dataset of cnns and a larger resnet dataset. For the latter, `sample_finetune_cifar100_resnet18.py` contains an example for model sampling. As above, set the path to a pretrained SANE model and epoch. Then, run
```bash
python3 sample_finetune_cifar100_resnet18.py
```
within `experiments/resnet18-cifar100/`. Generating models requires a pre-processed CIFAR100 dataset that can be generated by running
```bash
python3 prepare_cifar100_dataset.py
```
The small cnn sample experiment code is structured correspondingly.

## Contact
Feel free to get in touch with us with any questions on the project or to request access to data and / or pretrained models. Reach out to `konstantin.schuerholt@unisg.ch`.

## Citation
If you use this code in your research, please cite our paper:
```
@inproceedings{schuerholt2024sane,
    title={Towards Scalable and Versatile Weight Space Learning},
    author={Konstantin Sch{"u}rholt and Michael W. Mahoney and Damian Borth},
    booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)},
    year={2024},
    organization={PMLR}
}
```

## License
This project is licensed under the MIT License.

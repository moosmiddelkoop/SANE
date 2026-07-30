#!/bin/bash
#SBATCH --job-name=preprocess_cnn_cifar10_halfepoch_93tok
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=logs/preprocess_cnn_halfepoch_93tok_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

# Activate environment (adjust path if you submit from a different directory)
source "$HOME/SANE/.venv/bin/activate"

cd "$HOME/SANE/data"
python preprocess_dataset_cnn_cifar10.py

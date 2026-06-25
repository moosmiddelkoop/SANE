#!/bin/bash
#SBATCH --job-name=pretrain_sane_mnist_smallcnnzoo
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretrain_sane_mnist_smallcnnzoo_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/smallcnnzoo-mnist"
mkdir -p logs

# Activate environment (adjust path if you submit from a different directory)
source "$HOME/SANE/.venv/bin/activate"

python pretrain_sane_mnist_smallcnnzoo.py

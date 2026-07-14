#!/bin/bash
#SBATCH --job-name=pretrain_sane_cifar10_smallcnnzoo
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=6:00:00
#SBATCH --output=logs/pretrain_sane_cifar10_smallcnnzoo_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

set -euo pipefail

cd "$HOME/SANE/experiments/smallcnnzoo-cifar10"
mkdir -p logs

# consolidated dataset: single dataset.pt with stacked tensors, loaded into RAM
# once at startup (no per-sample file reads, no staging copy needed --
# /scratch-local is GPFS-backed on these nodes anyway, not local disk)
export SANE_DATA_DIR="/projects/prjs2156/shared/wsl/unthi_zoo/unthi_cifar10_preprocessed/consolidated"

# Activate environment (adjust path if you submit from a different directory)
source "$HOME/SANE/.venv/bin/activate"

python pretrain_sane_cifar10_smallcnnzoo.py

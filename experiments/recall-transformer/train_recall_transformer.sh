#!/bin/bash
#SBATCH --job-name=recall_transformer
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/recall-transformer/logs/train_recall_transformer_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch train_recall_transformer.sh <zoo: mnist, fmnist, cifar10 or svhn> [init: scratch (default) or sane] [seed: 0]
# needs unthi_<zoo>_preprocessed/tokens_ep4-8.pt from: sbatch build_tokens.sh <zoo>

set -euo pipefail

cd "$HOME/SANE/experiments/recall-transformer"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

python train_recall_transformer.py --zoo="$1" --init="${2:-scratch}" --seed="${3:-0}"

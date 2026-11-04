#!/bin/bash
#SBATCH --job-name=recall_transformer_fmnist
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/home1/mmiddelkoop/SANE/experiments/recall-transformer/logs/train_recall_transformer_fmnist_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=moos@cwi.nl

# usage: sbatch train_recall_transformer_fmnist.sh [init: scratch (default) or sane] [seed: 0]
# needs unthi_fmnist_preprocessed/tokens_ep4-8.pt from: sbatch build_tokens.sh fmnist

set -euo pipefail

cd "$HOME/SANE/experiments/recall-transformer"
mkdir -p logs
source "$HOME/SANE/.venv/bin/activate"

python train_recall_transformer_fmnist.py --init="${1:-scratch}" --seed="${2:-0}"
